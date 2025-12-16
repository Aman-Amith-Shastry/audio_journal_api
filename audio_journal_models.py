import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from load_model import load_audio_waveform_from_url
from get_metrics import compute_speech_metrics
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="aman-shastry/soundmind_model",
    filename="best_acc.pt"
)

state_dict = torch.load(model_path, map_location="cpu")


wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")

class WavLMForEmotion(nn.Module):
    def __init__(self, num_labels=7, freeze_feature_extractor=False):
        super().__init__()
        self.wavlm = wavlm
        hidden = self.wavlm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )
        if freeze_feature_extractor:
            for param in self.wavlm.parameters():
                param.requires_grad = False

    def forward(self, inputs, attention):
        ip = {'input_values': inputs,
              'attention_mask': attention}
        outputs = self.wavlm(**ip)
        # mean pooling over time
        hidden_states = outputs.last_hidden_state  # [batch, time, hidden]
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

model = WavLMForEmotion()
# Load the state_dict into the model instance
model.load_state_dict(state_dict)

model.to(device)
# Set the model to evaluation mode (important for inference)
model.eval()

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
# feature_extractor.to(device)

print("Loaded feature extractor")

def preprocess(path):
    # Load audio
    y = load_audio_waveform_from_url(path)
    # Optional: trim silence, augment here if you like
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    example = dict()
    example["input_values"] = torch.Tensor(inputs["input_values"][0].numpy()).unsqueeze(0)
    example["attention_mask"] = torch.Tensor(inputs["attention_mask"][0].numpy()).unsqueeze(0)
    return example, compute_speech_metrics(y)
