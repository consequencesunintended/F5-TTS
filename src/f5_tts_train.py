# file: f5_tts_train.py
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
import wandb
import gc
import torchaudio
from datasets import load_dataset

# Import your F5-TTS modules
from f5_tts.model.modules import MelSpec
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

def collate_fn(batch):
    import torch.nn.functional as F
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)
    mel_specs = torch.stack(padded_mel_specs)

    # For text, here we simply convert string to list of characters.
    # You may want to perform tokenization and padding.
    text = [
        list(item["text"]) if isinstance(item["text"], str) else item["text"]
        for item in batch
    ]
    text_lengths = torch.LongTensor([len(t) for t in text])
    return {
        "mel": mel_specs,
        "mel_lengths": mel_lengths,
        "text": text,
        "text_lengths": text_lengths,
    }

class HFDataset(IterableDataset):
    def __init__(self, hf_dataset, target_sample_rate=24000, n_mel_channels=100,
                 hop_length=256, n_fft=1024, win_length=1024, mel_spec_type="vocos"):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def __len__(self):
        return 3387817

    def __iter__(self):
        for row in self.data:
            audio = row["mp3"]["array"]
            sample_rate = row["mp3"]["sampling_rate"]
            duration = audio.shape[-1] / sample_rate
            if duration > 30 or duration < 0.3:
                continue
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                audio = resampler(audio)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            mel_spec = self.mel_spectrogram(audio).squeeze(0)
            text = row["json"]["text"]
            yield {"mel_spec": mel_spec, "text": text}

def main():
    from accelerate import Accelerator, DataLoaderConfiguration

    # Create a configuration for DataLoader settings
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=False,  # Each process fetches its own batch
        split_batches=True       # Split fetched batches across processes (if needed)
    )

    # Initialize Accelerator with the dataloader configuration
    accelerator = Accelerator(dataloader_config=dataloader_config)
       
    if accelerator.is_main_process:
        wandb.init(project="emilia")
    accelerator.wait_for_everyone()

    train_dataset = HFDataset(
        load_dataset("amphion/Emilia-Dataset", data_dir="Emilia/EN", split="train", streaming=True).with_format("torch")
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=3,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    target_sample_rate = 24000
    n_mel_channels = 100
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    vocab_char_map, vocab_size = get_tokenizer("amphion_Emilia-Dataset", "pinyin")
    mel_spec_kwargs = {
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": n_mel_channels,
        "target_sample_rate": target_sample_rate,
        "mel_spec_type": "vocos",
    }
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    optimizer = AdamW(model.parameters(), lr=1e-5)
    warmup_updates = 108
    total_updates = 100000  # adjust as needed
    decay_updates = total_updates - warmup_updates
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                             milestones=[warmup_updates])
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    model.train()
    for step, batch in enumerate(train_dataloader):
        gc.collect()
        torch.cuda.empty_cache()
        mel_spec = batch["mel"].permute(0, 2, 1)
        mel_lengths = batch["mel_lengths"]
        text_inputs = batch["text"]
        loss, cond, pred = model(mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=None)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # Print GPU memory usage
        current_alloc = torch.cuda.memory_allocated() / 1e9
        max_alloc = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[Process {accelerator.process_index} - GPU {accelerator.device}] "
            f"Step {step} | Loss: {loss.item():.4f} | Current: {current_alloc:.2f}GB, Peak: {max_alloc:.2f}GB"
        )
        if accelerator.is_main_process and step % 100 == 0:
            wandb.log({"step": step, "loss": loss.item(), "gpu_mem": current_alloc})
        if accelerator.is_main_process and step % 10000 == 0 and step > 0:
            ckpt_name = f'model_checkpoint_step{step}.pth'
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_name)
            wandb.save(ckpt_name)
        if step == 50000:
            break
    final_checkpoint_name = "final_checkpoint.pth"
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), final_checkpoint_name)
        wandb.save(final_checkpoint_name)
        wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    import sys
    # Make sure Python sees our cloned F5-TTS code
    # so we can import f5_tts.* directly
    sys.path.append("/F5-TTS/src")
    main()
