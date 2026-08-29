import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import timm


class MalimgDataset(Dataset):

    def __init__(self, root_dir, image_size=224):

        self.root_dir = root_dir
        self.image_size = image_size

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(self.classes)
        }

        self.samples = []

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"
        )

        for cls_name in self.classes:

            class_dir = os.path.join(
                root_dir,
                cls_name
            )

            for file_name in os.listdir(class_dir):

                if file_name.lower().endswith(
                    valid_extensions
                ):

                    file_path = os.path.join(
                        class_dir,
                        file_name
                    )

                    label = self.class_to_idx[
                        cls_name
                    ]

                    self.samples.append(
                        (file_path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size)
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1)

        return image, label

malimg_path = (
    "C:/Users/Legion/Malimg/"
    "malimg_paper_dataset_imgs"
)

dataset = MalimgDataset(
    malimg_path,
    image_size=224
)

print("Total samples :", len(dataset))
print("Number classes:", len(dataset.classes))
print("Classes       :", dataset.classes)



total_size = len(dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)

print("Training   :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Testing    :", len(test_dataset))


batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


patch_embed = PatchEmbed(
    img_size=224,
    patch_size=4,
    in_channels=3,
    embed_dim=128
).to(device)


mask_layer = RandomMasking(
    mask_ratio=0.60
).to(device)


mask_token_layer = MaskTokenInsertion(
    embed_dim=128
).to(device)

class MalVEITSwinEncoder(nn.Module):

    def __init__(self, input_dim=128):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            96
        )

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )

    def forward(self, x):

        B, N, D = x.shape

        # 128 → 96
        x = self.input_projection(x)

        # 3136 = 56 × 56
        x = x.reshape(
            B,
            56,
            56,
            96
        )

        # Swin stages
        x = self.swin.layers[0](x)
        x = self.swin.layers[1](x)
        x = self.swin.layers[2](x)
        x = self.swin.layers[3](x)

        x = self.swin.norm(x)

        return x


swin_encoder = MalVEITSwinEncoder(
    input_dim=128
).to(device)

class VAEBottleneck(nn.Module):

    def __init__(
        self,
        input_dim=768,
        latent_dim=256
    ):

        super().__init__()

        self.fc_mu = nn.Linear(
            input_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            input_dim,
            latent_dim
        )

    def forward(self, x):

        # x = (B, 7, 7, 768)

        x = x.mean(
            dim=(1, 2)
        )

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        z = mu + eps * std

        return z, mu, logvar


vae_bottleneck = VAEBottleneck(
    input_dim=768,
    latent_dim=256
).to(device)


class VAEDecoder(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        output_channels=3
    ):

        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            7 * 7 * 256
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(
            z.size(0),
            256,
            7,
            7
        )

        reconstruction = self.decoder(x)

        return reconstruction


vae_decoder = VAEDecoder(
    latent_dim=256,
    output_channels=3
).to(device)


class ClassificationHead(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        num_classes=26
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                latent_dim,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, z):

        return self.classifier(z)


classifier = ClassificationHead(
    latent_dim=256,
    num_classes=26
).to(device)


images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

print("Input:", images.shape)

# Patch embedding
patches = patch_embed(images)

print("Patch embeddings:", patches.shape)

# Random masking
visible_patches, mask, ids_restore = mask_layer(
    patches
)

print("Visible patches:", visible_patches.shape)

# Mask token insertion
x_full = mask_token_layer(
    visible_patches,
    ids_restore
)

print("After mask token insertion:", x_full.shape)

# Swin
swin_features = swin_encoder(
    x_full
)

print("Swin features:", swin_features.shape)

# VAE bottleneck
z, mu, logvar = vae_bottleneck(
    swin_features
)

print("Latent z:", z.shape)

# Decoder
reconstruction = vae_decoder(z)

print("Reconstruction:", reconstruction.shape)

# Classifier
logits = classifier(z)

print("Classification logits:", logits.shape)


class MalVEIT(nn.Module):

    def __init__(self, num_classes=25):

        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=4,
            in_channels=3,
            embed_dim=128
        )

        self.masking = RandomMasking(
            mask_ratio=0.60
        )

        self.mask_token = MaskTokenInsertion(
            embed_dim=128
        )

        self.swin = MalVEITSwinEncoder(
            input_dim=128
        )

        self.vae = VAEBottleneck(
            input_dim=768,
            latent_dim=256
        )

        self.decoder = VAEDecoder(
            latent_dim=256,
            output_channels=3
        )

        self.classifier = ClassificationHead(
            latent_dim=256,
            num_classes=num_classes
        )

    def forward(self, x):

        # Patch Embedding
        patches = self.patch_embed(x)

        # Random Masking
        visible_patches, mask, ids_restore = self.masking(
            patches
        )

        # Mask Token Insertion
        x_full = self.mask_token(
            visible_patches,
            ids_restore
        )

        # Swin Transformer
        swin_features = self.swin(
            x_full
        )

        # VAE Bottleneck
        z, mu, logvar = self.vae(
            swin_features
        )

        # Reconstruction
        reconstruction = self.decoder(z)

        # Classification
        logits = self.classifier(z)

        return (
            logits,
            reconstruction,
            mu,
            logvar,
            mask
        )


model = MalVEIT(
    num_classes=25
).to(device)

print(model)


reconstruction_criterion = nn.MSELoss()

classification_criterion = nn.CrossEntropyLoss()

lambda1 = 1.0
lambda2 = 1.0

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


num_epochs = 100

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


for epoch in range(num_epochs):

    # ==================================================
    # TRAINING
    # ==================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        (
            logits,
            reconstruction,
            mu,
            logvar,
            mask
        ) = model(images)

        # ----------------------------------------------
        # Reconstruction Loss
        # ----------------------------------------------

        reconstruction_loss = reconstruction_criterion(
            reconstruction,
            images
        )

        # ----------------------------------------------
        # KL Loss
        # ----------------------------------------------

        kl_loss = kl_divergence_loss(
            mu,
            logvar
        )

        # ----------------------------------------------
        # Classification Loss
        # ----------------------------------------------

        classification_loss = classification_criterion(
            logits,
            labels
        )

        # ----------------------------------------------
        # VAE Loss
        # ----------------------------------------------

        vae_loss = (
            reconstruction_loss +
            kl_loss
        )

        # ----------------------------------------------
        # Total Loss
        # ----------------------------------------------

        total_loss = (
            lambda1 * vae_loss +
            lambda2 * classification_loss
        )

        # Backpropagation
        total_loss.backward()

        optimizer.step()

        # ----------------------------------------------
        # Statistics
        # ----------------------------------------------

        running_loss += (
            total_loss.item() *
            images.size(0)
        )

        _, predicted = torch.max(
            logits,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_train_loss = (
        running_loss / total
    )

    epoch_train_accuracy = (
        100.0 * correct / total
    )


    # ==================================================
    # VALIDATION
    # ==================================================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            (
                logits,
                reconstruction,
                mu,
                logvar,
                mask
            ) = model(images)

            # Reconstruction
            reconstruction_loss = reconstruction_criterion(
                reconstruction,
                images
            )

            # KL
            kl_loss = kl_divergence_loss(
                mu,
                logvar
            )

            # Classification
            classification_loss = classification_criterion(
                logits,
                labels
            )

            # Total
            vae_loss = (
                reconstruction_loss +
                kl_loss
            )

            total_loss = (
                lambda1 * vae_loss +
                lambda2 * classification_loss
            )

            val_running_loss += (
                total_loss.item() *
                images.size(0)
            )

            _, predicted = torch.max(
                logits,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    epoch_val_loss = (
        val_running_loss / val_total
    )

    epoch_val_accuracy = (
        100.0 * val_correct / val_total
    )


    # ==================================================
    # STORE RESULTS
    # ==================================================

    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )

    train_accuracies.append(
        epoch_train_accuracy
    )

    val_accuracies.append(
        epoch_val_accuracy
    )


    print(
        f"Epoch [{epoch+1:03d}/{num_epochs}] "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Train Acc: {epoch_train_accuracy:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val Acc: {epoch_val_accuracy:.2f}%"
    )

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# ============================================================
# 1. Select Adialer.C Malimg sample
# ============================================================

image_path = (
    "C:\\Users\\Legion\\Malimg\\"
    "malimg_paper_dataset_imgs\\Adialer.C\\"
    "000e30a0819ac7ed931d629ce2ef8948.png"
)


# ============================================================
# 2. Load and preprocess image
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")

input_tensor = (
    transform(image)
    .unsqueeze(0)
    .to(device)
)

print("Malware family : Adialer.C")
print("Input shape    :", input_tensor.shape)


# ============================================================
# 3. Feature-map hook
# ============================================================

activation = {}

def hook_fn(module, input, output):
    activation["feature_map"] = output.detach()


# ============================================================
# 4. Register hook on final Swin Transformer stage
# ============================================================

hook = model.swin.swin.layers[3].register_forward_hook(
    hook_fn
)


# ============================================================
# 5. Forward pass through MalVEIT
# ============================================================

model.eval()

with torch.no_grad():

    (
        logits,
        reconstruction,
        mu,
        logvar,
        mask
    ) = model(input_tensor)


# ============================================================
# 6. Get predicted class
# ============================================================

predicted_class = torch.argmax(
    logits,
    dim=1
).item()

probability = torch.softmax(
    logits,
    dim=1
)[0, predicted_class].item()

print("Predicted class index :", predicted_class)
print(
    "Prediction confidence:",
    f"{probability * 100:.2f}%"
)


# ============================================================
# 7. Get Swin feature map
# ============================================================

feature_map = activation["feature_map"]

print(
    "Original feature map shape:",
    feature_map.shape
)


# ============================================================
# 8. Convert feature map to
#    (Channels, Height, Width)
# ============================================================

if feature_map.dim() == 4:

    # Swin output:
    # (B, H, W, C)

    feature_map = feature_map.squeeze(0)

    feature_map = feature_map.permute(
        2, 0, 1
    )


elif feature_map.dim() == 3:

    # Swin output:
    # (B, N, C)

    feature_map = feature_map.squeeze(0)

    num_patches = feature_map.shape[0]

    spatial_size = int(
        np.sqrt(num_patches)
    )

    feature_map = feature_map.reshape(
        spatial_size,
        spatial_size,
        -1
    )

    feature_map = feature_map.permute(
        2, 0, 1
    )


print(
    "Feature map after conversion:",
    feature_map.shape
)


# ============================================================
# 9. Select first 16 feature maps
# ============================================================

num_channels = min(
    16,
    feature_map.shape[0]
)

selected_features = feature_map[
    :num_channels
]


# ============================================================
# 10. Normalize each feature map
# ============================================================

normalized_features = []

for i in range(num_channels):

    fmap = selected_features[i]

    fmap_min = fmap.min()
    fmap_max = fmap.max()

    fmap = (
        fmap - fmap_min
    ) / (
        fmap_max - fmap_min + 1e-8
    )

    normalized_features.append(
        fmap.cpu().numpy()
    )


# ============================================================
# 11. Create large high-quality figure
# ============================================================

fig, axes = plt.subplots(
    nrows=4,
    ncols=4,
    figsize=(18, 18),
    dpi=600
)


for i, ax in enumerate(axes.flat):

    if i < num_channels:

        ax.imshow(
            normalized_features[i],
            cmap="viridis",
            interpolation="nearest",
            aspect="equal"
        )

        # No individual feature labels
        ax.axis("off")

    else:

        ax.axis("off")


# ============================================================
# 12. Figure title
# ============================================================

fig.suptitle(
    "Feature Maps",
    fontsize=30,
    fontweight="bold",
    y=0.985
)


# ============================================================
# 13. Reduce unnecessary whitespace
# ============================================================

plt.subplots_adjust(
    left=0.015,
    right=0.985,
    bottom=0.015,
    top=0.94,
    wspace=0.04,
    hspace=0.04
)


# ============================================================
# 14. Save 600 DPI PNG
# ============================================================

fig.savefig(
    "Adialer_C_Feature_Maps_600dpi.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
    edgecolor="none"
)


# ============================================================
# 15. Save 1200 DPI PNG
# ============================================================

fig.savefig(
    "Adialer_C_Feature_Maps_1200dpi.png",
    dpi=1200,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
    edgecolor="none"
)


# ============================================================
# 16. Save PDF
# ============================================================

fig.savefig(
    "Adialer_C_Feature_Maps.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
    edgecolor="none"
)


# ============================================================
# 17. Display
# ============================================================

plt.show()


# ============================================================
# 18. Remove hook
# ============================================================

hook.remove()


print("\nFeature-map visualization completed.")
print("Malware family: Adialer.C")
print("300 DPI PNG  : Adialer_C_Feature_Maps_600dpi.png")
print("PDF          : Adialer_C_Feature_Maps.pdf")


import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap

from PIL import Image
import torchvision.transforms as transforms


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = model.to(device)
model.eval()


# ============================================================
# 2. C2LOP.gen!g directory
# ============================================================

c2lop_dir = (
    r"C:\Users\Legion\Malimg\malimg_paper_dataset_imgs"
    r"\C2LOP.gen!g"
)


# ============================================================
# 3. Target C2LOP.gen!g image
# ============================================================

target_image_path = (
    c2lop_dir +
    r"\000c2a21b0a32178fa7ac19c3b83b2e3.png"
)


print("\nTarget image:")
print(target_image_path)


# ============================================================
# 4. Find other C2LOP.gen!g images
# ============================================================

all_c2lop_images = [

    os.path.join(
        c2lop_dir,
        f
    )

    for f in os.listdir(c2lop_dir)

    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]


print(
    "\nTotal C2LOP.gen!g images found:",
    len(all_c2lop_images)
)


# ============================================================
# 5. Select background images
#
# IMPORTANT:
# The target image itself is excluded.
#
# Background consists ONLY of C2LOP.gen!g samples.
# ============================================================

background_paths = [

    p

    for p in all_c2lop_images

    if os.path.abspath(p)
    != os.path.abspath(target_image_path)
]


# Use at most 8 background images
background_paths = background_paths[:8]


print(
    "C2LOP background images:",
    len(background_paths)
)


for i, p in enumerate(
    background_paths
):

    print(
        f"Background {i+1}:",
        os.path.basename(p)
    )


# ============================================================
# 6. Image transformation
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224),
        antialias=True
    ),

    transforms.ToTensor()
])


# ============================================================
# 7. Load image
# ============================================================

def load_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    tensor = transform(
        image
    )

    return tensor


# ============================================================
# 8. Load C2LOP background
# ============================================================

background_images = []

for path in background_paths:

    img = load_image(
        path
    )

    background_images.append(
        img
    )


# Make sure background exists
if len(background_images) == 0:

    raise RuntimeError(
        "No background images were found in the C2LOP.gen!g folder."
    )


background = torch.stack(
    background_images
).to(device)


print(
    "\nBackground shape:",
    background.shape
)


# ============================================================
# 9. MalVEIT wrapper
#
# Returns classification logits only
# ============================================================

class MalVEITWrapper(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model


    def forward(self, x):

        output = self.model(x)

        # MalVEIT output:
        #
        # output[0] = classification logits
        # output[1] = reconstruction
        # output[2] = mu
        # output[3] = logvar
        # output[4] = mask

        logits = output[0]

        return logits


# ============================================================
# 10. Create SHAP model
# ============================================================

shap_model = MalVEITWrapper(
    model
)

shap_model.eval()


# ============================================================
# 11. Create SHAP Gradient Explainer
# ============================================================

print(
    "\nCreating SHAP explainer..."
)

explainer = shap.GradientExplainer(
    shap_model,
    background
)

print(
    "SHAP explainer ready."
)


# ============================================================
# 12. Load C2LOP image
# ============================================================

image_tensor = load_image(
    target_image_path
)


input_tensor = image_tensor.unsqueeze(
    0
).to(device)


print(
    "\nInput shape:",
    input_tensor.shape
)


# ============================================================
# 13. Prediction
# ============================================================

with torch.no_grad():

    logits = shap_model(
        input_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[
            0,
            predicted_class
        ].item()
        * 100
    )


print(
    "\nC2LOP.gen!g prediction"
)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence:.2f}%"
)


# ============================================================
# 14. Generate SHAP values
# ============================================================

print(
    "\nCalculating SHAP values..."
)

shap_values = explainer.shap_values(
    input_tensor,
    nsamples=100
)


# ============================================================
# 15. Inspect SHAP output
# ============================================================

print(
    "\nSHAP output type:",
    type(shap_values)
)


if isinstance(
    shap_values,
    list
):

    print(
        "Number of output classes:",
        len(shap_values)
    )

    values = shap_values[
        predicted_class
    ]

else:

    values = shap_values


values = np.asarray(
    values
)


print(
    "Raw SHAP shape:",
    values.shape
)


# ============================================================
# 16. Correct SHAP dimension handling
# ============================================================

# Expected possibilities include:
#
# (1, 3, 224, 224)
# (1, 224, 224, 3)
# (1, 1, 3, 224, 224)
# etc.

while values.ndim > 4:

    values = values[0]


if values.ndim == 4:

    # Remove batch dimension

    values = values[0]


print(
    "SHAP shape after processing:",
    values.shape
)


# ============================================================
# 17. Convert RGB SHAP values
#     into pixel-wise importance
# ============================================================

if values.ndim != 3:

    raise ValueError(
        "Unexpected SHAP shape: "
        + str(values.shape)
    )


# ------------------------------------------------------------
# Case 1: H x W x C
# ------------------------------------------------------------

if values.shape[-1] == 3:

    pixel_importance = np.mean(
        np.abs(values),
        axis=-1
    )


# ------------------------------------------------------------
# Case 2: C x H x W
# ------------------------------------------------------------

elif values.shape[0] == 3:

    pixel_importance = np.mean(
        np.abs(values),
        axis=0
    )


else:

    raise ValueError(
        "Unable to identify RGB dimension: "
        + str(values.shape)
    )


print(
    "Pixel importance shape:",
    pixel_importance.shape
)


# ============================================================
# 18. Robust normalization
# ============================================================

low = np.percentile(
    pixel_importance,
    1
)

high = np.percentile(
    pixel_importance,
    99
)


pixel_importance = np.clip(
    pixel_importance,
    low,
    high
)


if high > low:

    pixel_importance = (

        pixel_importance - low

    ) / (

        high - low

    )

else:

    pixel_importance = np.zeros_like(
        pixel_importance
    )


# ============================================================
# 19. Enhance visibility
# ============================================================

pixel_importance = np.power(
    pixel_importance,
    0.60
)


# ============================================================
# 20. Create publication-quality figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7),
    dpi=600
)


ax.imshow(
    pixel_importance,
    cmap="jet",
    vmin=0,
    vmax=1,
    interpolation="bilinear",
    aspect="equal"
)


# ============================================================
# 21. Remove axes
# ============================================================

ax.set_xticks([])
ax.set_yticks([])


for spine in ax.spines.values():

    spine.set_visible(False)


# ============================================================
# 22. Panel label
# ============================================================

ax.text(
    0.5,
    -0.035,
    "(a)",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=22,
    fontweight="bold"
)


# ============================================================
# 23. Reduce whitespace
# ============================================================

plt.subplots_adjust(
    left=0.01,
    right=0.99,
    top=0.99,
    bottom=0.09
)


# ============================================================
# 24. Save PNG - 600 DPI
# ============================================================

fig.savefig(
    "C2LOP_SHAP_Heatmap_600dpi.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    facecolor="white",
    edgecolor="none"
)


# ============================================================
# 25. Save PDF
# ============================================================

fig.savefig(
    "C2LOP_SHAP_Heatmap.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
    facecolor="white",
    edgecolor="none"
)


# ============================================================
# 26. Display
# ============================================================

plt.show()


# ============================================================
# 27. Final information
# ============================================================

print("\n==============================================")
print("C2LOP.gen!g SHAP visualization completed")
print("==============================================")

print(
    "Target:",
    os.path.basename(
        target_image_path
    )
)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence:.2f}%"
)

print(
    "PNG: C2LOP_SHAP_Heatmap_300dpi.png"
)

print(
    "PDF: C2LOP_SHAP_Heatmap.pdf"
)




