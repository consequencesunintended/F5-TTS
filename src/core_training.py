# core_training.py
import os
import gc
import wandb
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

# Import modules from F5-TTS (make sure the F5-TTS repo is cloned and in PYTHONPATH)
from f5_tts.model.modules import MelSpec
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

# Define the dataset class
class HFDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset,
        target_sample_rate=24000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
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
        # This is a placeholder. Replace with your dataset length if known.
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

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)

            text = row["json"]["text"]
            yield {"mel_spec": mel_spec, "text": text}


def train():
    print("Current working directory:", os.getcwd())
    wandb.init(project="emilia")

    # Create dataset and dataloader
    train_dataset = HFDataset(
        load_dataset(
            "amphion/Emilia-Dataset",
            data_dir="Emilia/EN",
            split="train",
            use_auth_token=os.environ["HF_TOKEN"],
            streaming=True,
        ).with_format("torch")
    )

    def collate_fn(batch):
        mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = torch.nn.functional.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        text = [
            list(item["text"]) if isinstance(item["text"], str) else item["text"]
            for item in batch
        ]
        text_lengths = torch.LongTensor([len(t) for t in text])

        return dict(
            mel=mel_specs,
            mel_lengths=mel_lengths,
            text=text,
            text_lengths=text_lengths,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=6,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model and optimizer setup
    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    mel_spec_type = "vocos"

    # Define model class and configuration
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    vocab_char_map, vocab_size = get_tokenizer("amphion_Emilia-Dataset", "pinyin")

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    ).to("cuda")

    optimizer = AdamW(model.parameters(), lr=1e-5)

    warmup_updates = 108
    total_updates = len(train_dataloader)
    decay_updates = total_updates - warmup_updates
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
    )

    model.train()

    # Training loop
    for step, batch in enumerate(train_dataloader):
        gc.collect()
        torch.cuda.empty_cache()

        text_inputs = batch["text"]

        mel_spec = batch["mel"].permute(0, 2, 1).to("cuda")
        mel_lengths = batch["mel_lengths"].to("cuda")

        loss, cond, pred = model(
            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=None
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(f"step:{step}  loss:{loss.item()}")

        if step % 10000 == 0:
            torch.save(model.state_dict(), f'model_checkpoint_epoch{step+1}.pth')
            wandb.save(f'model_checkpoint_epoch{step+1}.pth')

        if step == 50000:
            break

    final_checkpoint_name = "final_checkpoint.pth"
    torch.save(model.state_dict(), final_checkpoint_name)
    wandb.save(final_checkpoint_name)
    wandb.finish()

    return "Training completed!"
