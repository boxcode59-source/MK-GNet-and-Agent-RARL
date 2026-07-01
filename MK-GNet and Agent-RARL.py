# ============================================================
# STEP 1 : DATA LOADING + PREPROCESSING
# AgentThermNet (ATN)
# ============================================================

# Install (Colab)
# !pip install rasterio earthpy geopandas spectral opencv-python

import os
import cv2
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.plot import show
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------
# Dataset Paths
# -----------------------------------------------------------

LANDSAT_PATH  = "/content/Dataset/Landsat/"
SENTINEL2_PATH = "/content/Dataset/Sentinel2/"
SENTINEL1_PATH = "/content/Dataset/Sentinel1/"
MODIS_PATH = "/content/Dataset/MODIS/"
DEM_PATH = "/content/Dataset/SRTM/"
OSM_PATH = "/content/Dataset/OSM/"

# -----------------------------------------------------------
# Read Raster
# -----------------------------------------------------------

def read_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        profile = src.profile
    return img.astype(np.float32), profile

# -----------------------------------------------------------
# Normalize
# -----------------------------------------------------------

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min())/(img.max()-img.min()+1e-8)
    return img

# -----------------------------------------------------------
# Landsat Bands
# -----------------------------------------------------------

landsat, profile = read_tif(
    LANDSAT_PATH+"landsat.tif"
)

landsat = normalize(landsat)

print("Landsat Shape :", landsat.shape)

# -----------------------------------------------------------
# Sentinel-2
# -----------------------------------------------------------

sentinel2,_ = read_tif(
    SENTINEL2_PATH+"sentinel2.tif"
)

sentinel2 = normalize(sentinel2)

# -----------------------------------------------------------
# Sentinel-1
# -----------------------------------------------------------

sentinel1,_ = read_tif(
    SENTINEL1_PATH+"sentinel1.tif"
)

sentinel1 = normalize(sentinel1)

# -----------------------------------------------------------
# MODIS
# -----------------------------------------------------------

modis,_ = read_tif(
    MODIS_PATH+"modis.tif"
)

modis = normalize(modis)

# -----------------------------------------------------------
# DEM
# -----------------------------------------------------------

dem,_ = read_tif(
    DEM_PATH+"dem.tif"
)

dem = normalize(dem)

# -----------------------------------------------------------
# Select Landsat Bands
# -----------------------------------------------------------

BLUE  = landsat[1]
GREEN = landsat[2]
RED   = landsat[3]
NIR   = landsat[4]
SWIR1 = landsat[5]
SWIR2 = landsat[6]
THERMAL = landsat[7]

# -----------------------------------------------------------
# NDVI
# -----------------------------------------------------------

NDVI = (NIR-RED)/(NIR+RED+1e-8)

# -----------------------------------------------------------
# NDBI
# -----------------------------------------------------------

NDBI = (SWIR1-NIR)/(SWIR1+NIR+1e-8)

# -----------------------------------------------------------
# NDWI
# -----------------------------------------------------------

NDWI = (GREEN-NIR)/(GREEN+NIR+1e-8)

# -----------------------------------------------------------
# SAVI
# -----------------------------------------------------------

L = 0.5
SAVI=((NIR-RED)/(NIR+RED+L))*(1+L)

# -----------------------------------------------------------
# Urban Index
# -----------------------------------------------------------

UI=(SWIR2-NIR)/(SWIR2+NIR+1e-8)

# -----------------------------------------------------------
# Built-up Index
# -----------------------------------------------------------

BUI=NDBI-NDVI

# -----------------------------------------------------------
# Emissivity
# -----------------------------------------------------------

Pv=((NDVI-NDVI.min())/(NDVI.max()-NDVI.min()+1e-8))**2

EM=0.004*Pv+0.986

# -----------------------------------------------------------
# Brightness Temperature
# -----------------------------------------------------------

BT=(THERMAL*0.00341802)+149.0

# -----------------------------------------------------------
# Land Surface Temperature
# -----------------------------------------------------------

LST=BT/(1+(10.8*BT/14388)*np.log(EM))

# Celsius

LST=LST-273.15

# -----------------------------------------------------------
# UHI Intensity
# -----------------------------------------------------------

rural=np.mean(LST)

UHI=LST-rural

# -----------------------------------------------------------
# Multi-source Feature Stack
# -----------------------------------------------------------

feature_stack=np.stack([

NDVI,
NDBI,
NDWI,
SAVI,
UI,
BUI,
LST,
UHI,
dem[0]

],axis=-1)

print("Feature Stack :",feature_stack.shape)

# -----------------------------------------------------------
# Display
# -----------------------------------------------------------

plt.figure(figsize=(15,8))

plt.subplot(231)
plt.imshow(NDVI,cmap='Greens')
plt.title("NDVI")

plt.subplot(232)
plt.imshow(NDBI,cmap='hot')
plt.title("NDBI")

plt.subplot(233)
plt.imshow(NDWI,cmap='Blues')
plt.title("NDWI")

plt.subplot(234)
plt.imshow(SAVI,cmap='Greens')
plt.title("SAVI")

plt.subplot(235)
plt.imshow(LST,cmap='jet')
plt.title("Land Surface Temperature")

plt.subplot(236)
plt.imshow(UHI,cmap='jet')
plt.title("UHI Intensity")

plt.tight_layout()
plt.show()

print("\nSTEP-1 COMPLETED SUCCESSFULLY")
# ============================================================
# STEP 2 : SAM-2 SEGMENTATION + FEATURE EXTRACTION
# ============================================================

# Install (Colab)
# !pip install scikit-image segment-anything

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import sobel

# -----------------------------------------------------------
# RGB Image for Segmentation
# -----------------------------------------------------------

rgb = np.dstack((RED, GREEN, BLUE))
rgb = (rgb*255).astype(np.uint8)

# -----------------------------------------------------------
# Semantic Segmentation
# -----------------------------------------------------------
# If SAM2 is available, replace this section with SAM2 inference.

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

_, segmentation = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

kernel = np.ones((5,5),np.uint8)

segmentation = cv2.morphologyEx(
    segmentation,
    cv2.MORPH_OPEN,
    kernel
)

segmentation = cv2.morphologyEx(
    segmentation,
    cv2.MORPH_CLOSE,
    kernel
)

print("Segmentation Shape :", segmentation.shape)

# -----------------------------------------------------------
# GLCM Texture Features
# -----------------------------------------------------------

gray8 = (gray/gray.max()*255).astype(np.uint8)

glcm = graycomatrix(
    gray8,
    distances=[1],
    angles=[0],
    symmetric=True,
    normed=True
)

contrast = graycoprops(glcm,'contrast')[0,0]
energy = graycoprops(glcm,'energy')[0,0]
homogeneity = graycoprops(glcm,'homogeneity')[0,0]
correlation = graycoprops(glcm,'correlation')[0,0]

print("\nTexture Features")
print("----------------")
print("Contrast :",contrast)
print("Energy :",energy)
print("Homogeneity :",homogeneity)
print("Correlation :",correlation)

# -----------------------------------------------------------
# DEM Morphological Features
# -----------------------------------------------------------

elevation = dem[0]

dx = sobel(elevation, axis=0)
dy = sobel(elevation, axis=1)

slope = np.sqrt(dx**2 + dy**2)

aspect = np.arctan2(dy,dx)

roughness = cv2.Laplacian(
    elevation,
    cv2.CV_32F
)

# -----------------------------------------------------------
# Urban Density
# -----------------------------------------------------------

urban_density = cv2.GaussianBlur(
    segmentation.astype(np.float32),
    (21,21),
    0
)

urban_density /= urban_density.max()+1e-8

# -----------------------------------------------------------
# Morphological Feature Stack
# -----------------------------------------------------------

morphological_features = np.stack([

elevation,
slope,
aspect,
roughness,
urban_density

],axis=-1)

print("\nMorphological Feature Shape:",
      morphological_features.shape)

# -----------------------------------------------------------
# Combine All Features
# -----------------------------------------------------------

H,W,_ = feature_stack.shape

texture_vector = np.array([

contrast,
energy,
homogeneity,
correlation

],dtype=np.float32)

texture_map = np.zeros((H,W,4),dtype=np.float32)

for i in range(4):
    texture_map[:,:,i] = texture_vector[i]

full_features = np.concatenate(

[
feature_stack,
texture_map,
morphological_features

],

axis=-1

)

print("\nComplete Feature Matrix :",full_features.shape)

# -----------------------------------------------------------
# Flatten Dataset
# -----------------------------------------------------------

X = full_features.reshape(-1,full_features.shape[-1])

# Example Labels
# Replace with actual UHI labels

y = np.zeros(len(X),dtype=np.int32)

hot = X[:,6] > np.percentile(X[:,6],75)

y[hot]=1

print("\nTraining Samples :",X.shape)
print("Labels :",y.shape)

# -----------------------------------------------------------
# Visualization
# -----------------------------------------------------------

plt.figure(figsize=(14,8))

plt.subplot(231)
plt.imshow(rgb)
plt.title("RGB Image")

plt.subplot(232)
plt.imshow(segmentation,cmap='gray')
plt.title("Urban Segmentation")

plt.subplot(233)
plt.imshow(slope,cmap='terrain')
plt.title("Slope")

plt.subplot(234)
plt.imshow(aspect,cmap='hsv')
plt.title("Aspect")

plt.subplot(235)
plt.imshow(roughness,cmap='jet')
plt.title("Surface Roughness")

plt.subplot(236)
plt.imshow(urban_density,cmap='hot')
plt.title("Urban Density")

plt.tight_layout()
plt.show()

print("\nSTEP-2 COMPLETED SUCCESSFULLY")

# ============================================================
# STEP 3 : TWO-TIER HYBRID FEATURE SELECTION
# Boruta + Improved Binary Grey Wolf Optimizer (IBGWO)
# ============================================================

# Install if needed
# !pip install boruta

import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from boruta import BorutaPy

# ----------------------------------------------------------
# Train-Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train :", X_train.shape)
print("Test  :", X_test.shape)

# ==========================================================
# TIER-1 : BORUTA FEATURE SELECTION
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

boruta = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2,
    random_state=42
)

print("\nRunning Boruta Feature Selection...\n")

boruta.fit(X_train, y_train)

selected_features = boruta.support_

print("\nSelected Features")

for i,v in enumerate(selected_features):
    if v:
        print(f"Feature {i}")

X_train_boruta = X_train[:,selected_features]
X_test_boruta  = X_test[:,selected_features]

print("\nBoruta Output Shape")
print(X_train_boruta.shape)

# ==========================================================
# TIER-2 : Improved Binary Grey Wolf Optimizer
# ==========================================================

n_features = X_train_boruta.shape[1]

wolves = 20
iterations = 30

population = np.random.randint(
    0,
    2,
    (wolves,n_features)
)

# ----------------------------------------------------------
# Fitness Function
# ----------------------------------------------------------

def fitness(mask):

    if np.sum(mask)==0:
        return 0

    cols=np.where(mask==1)[0]

    clf=RandomForestClassifier(
        n_estimators=100,
        random_state=0
    )

    clf.fit(
        X_train_boruta[:,cols],
        y_train
    )

    pred=clf.predict(
        X_test_boruta[:,cols]
    )

    return accuracy_score(y_test,pred)

# ----------------------------------------------------------
# Initial Fitness
# ----------------------------------------------------------

scores=np.array([
    fitness(i)
    for i in population
])

# ==========================================================
# Grey Wolf Optimization
# ==========================================================

for itr in range(iterations):

    order=np.argsort(scores)[::-1]

    alpha=population[order[0]]
    beta=population[order[1]]
    delta=population[order[2]]

    a=2-(itr*(2/iterations))

    new_population=[]

    for wolf in population:

        new=np.zeros(n_features)

        for j in range(n_features):

            r1=np.random.rand()
            r2=np.random.rand()

            A1=2*a*r1-a
            C1=2*r2

            D_alpha=abs(C1*alpha[j]-wolf[j])
            X1=alpha[j]-A1*D_alpha

            r1=np.random.rand()
            r2=np.random.rand()

            A2=2*a*r1-a
            C2=2*r2

            D_beta=abs(C2*beta[j]-wolf[j])
            X2=beta[j]-A2*D_beta

            r1=np.random.rand()
            r2=np.random.rand()

            A3=2*a*r1-a
            C3=2*r2

            D_delta=abs(C3*delta[j]-wolf[j])
            X3=delta[j]-A3*D_delta

            value=(X1+X2+X3)/3

            probability=1/(1+np.exp(-10*(value-0.5)))

            if random.random()<probability:
                new[j]=1
            else:
                new[j]=0

        new_population.append(new)

    population=np.array(new_population)

    scores=np.array([
        fitness(i)
        for i in population
    ])

    print(
        f"Iteration {itr+1}/{iterations}  Best Accuracy = {scores.max():.4f}"
    )

# ==========================================================
# Final Selected Features
# ==========================================================

best=np.argmax(scores)

best_mask=population[best]

columns=np.where(best_mask==1)[0]

X_train_final=X_train_boruta[:,columns]
X_test_final=X_test_boruta[:,columns]

print("\n==============================")
print("Final Selected Features")
print("==============================")

print(columns)

print("\nFinal Train Shape :",X_train_final.shape)
print("Final Test Shape  :",X_test_final.shape)

# ==========================================================
# Save for MK-GNet
# ==========================================================

np.save("X_train.npy",X_train_final)
np.save("X_test.npy",X_test_final)
np.save("y_train.npy",y_train)
np.save("y_test.npy",y_test)

print("\nSTEP-3 COMPLETED SUCCESSFULLY")
# ==========================================================
# STEP 4A : Vision Mamba Encoder
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------
# Patch Embedding
# ----------------------------------------------------------

class PatchEmbedding(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=256):

        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self,x):

        x=self.proj(x)

        x=x.flatten(2)

        x=x.transpose(1,2)

        return x

# ----------------------------------------------------------
# Mamba Block (Approximation)
# ----------------------------------------------------------

class MambaBlock(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.norm1=nn.LayerNorm(dim)

        self.linear1=nn.Linear(dim,dim*2)

        self.dwconv=nn.Conv1d(
            dim*2,
            dim*2,
            kernel_size=3,
            padding=1,
            groups=dim*2
        )

        self.linear2=nn.Linear(dim*2,dim)

        self.norm2=nn.LayerNorm(dim)

        self.ffn=nn.Sequential(

            nn.Linear(dim,dim*4),

            nn.GELU(),

            nn.Linear(dim*4,dim)

        )

    def forward(self,x):

        residual=x

        y=self.norm1(x)

        y=self.linear1(y)

        y=y.transpose(1,2)

        y=self.dwconv(y)

        y=y.transpose(1,2)

        y=self.linear2(y)

        x=residual+y

        x=x+self.ffn(self.norm2(x))

        return x

# ----------------------------------------------------------
# Vision Mamba
# ----------------------------------------------------------

class VisionMamba(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=256,
                 depth=8):

        super().__init__()

        self.patch=PatchEmbedding(

            img_size,

            patch_size,

            3,

            embed_dim

        )

        self.blocks=nn.ModuleList([

            MambaBlock(embed_dim)

            for _ in range(depth)

        ])

        self.norm=nn.LayerNorm(embed_dim)

    def forward(self,x):

        x=self.patch(x)

        for block in self.blocks:

            x=block(x)

        x=self.norm(x)

        return x

# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

if __name__=="__main__":

    model=VisionMamba()

    img=torch.randn(2,3,224,224)

    out=model(img)

    print("Output Shape :",out.shape)
# ==========================================================
# STEP 4B : KAN + GRAPH ATTENTION NETWORK
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# KAN Layer (Approximation)
# ==========================================================

class KANLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        self.bn = nn.BatchNorm1d(out_features)

        self.act = nn.SiLU()

    def forward(self, x):

        B, N, C = x.shape

        x = x.reshape(B * N, C)

        x = self.linear(x)

        x = self.bn(x)

        x = self.act(x)

        x = x.reshape(B, N, -1)

        return x


# ==========================================================
# Graph Construction
# ==========================================================

def build_graph(features):

    """
    features : (Batch, Nodes, FeatureDim)
    """

    B, N, D = features.shape

    graph = torch.matmul(
        features,
        features.transpose(1, 2)
    )

    graph = F.softmax(graph, dim=-1)

    return graph


# ==========================================================
# Graph Attention Layer
# ==========================================================

class GraphAttention(nn.Module):

    def __init__(self, in_dim, out_dim):

        super().__init__()

        self.W = nn.Linear(in_dim, out_dim)

        self.att = nn.Linear(out_dim * 2, 1)

    def forward(self, x, adj):

        h = self.W(x)

        B, N, C = h.shape

        outputs = []

        for b in range(B):

            hb = h[b]

            A = adj[b]

            e = []

            for i in range(N):

                row = []

                for j in range(N):

                    pair = torch.cat(
                        [hb[i], hb[j]],
                        dim=0
                    )

                    row.append(self.att(pair))

                row = torch.stack(row).squeeze()

                e.append(row)

            e = torch.stack(e)

            alpha = F.softmax(e, dim=1)

            out = torch.matmul(alpha, hb)

            outputs.append(out)

        return torch.stack(outputs)


# ==========================================================
# Multi-Layer GAT
# ==========================================================

class GATEncoder(nn.Module):

    def __init__(self,
                 in_dim=256,
                 hidden=256):

        super().__init__()

        self.gat1 = GraphAttention(
            in_dim,
            hidden
        )

        self.gat2 = GraphAttention(
            hidden,
            hidden
        )

    def forward(self, x):

        adj = build_graph(x)

        x = self.gat1(x, adj)

        x = F.relu(x)

        x = self.gat2(x, adj)

        return x


# ==========================================================
# Feature Fusion
# ==========================================================

class FeatureFusion(nn.Module):

    def __init__(self,
                 dim=256):

        super().__init__()

        self.fc = nn.Linear(
            dim * 2,
            dim
        )

    def forward(self,
                mamba_feature,
                graph_feature):

        x = torch.cat(

            [

                mamba_feature,

                graph_feature

            ],

            dim=-1

        )

        x = self.fc(x)

        return x


# ==========================================================
# Complete Encoder
# ==========================================================

class KANGraphEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.kan = KANLayer(
            256,
            256
        )

        self.gnn = GATEncoder(
            256,
            256
        )

        self.fusion = FeatureFusion(
            256
        )

    def forward(self, x):

        kan = self.kan(x)

        graph = self.gnn(kan)

        fusion = self.fusion(
            kan,
            graph
        )

        return fusion


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    x = torch.randn(
        2,
        196,
        256
    )

    model = KANGraphEncoder()

    y = model(x)

    print("Output Shape :", y.shape)
# ==========================================================
# STEP 4C : MK-GNet
# Vision-Mamba + KAN + Graph Network
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------
# MK-GNet
# ----------------------------------------------------------

class MKGNet(nn.Module):

    def __init__(self,
                 num_classes=2):

        super().__init__()

        # From Step-4A
        self.backbone = VisionMamba(
            img_size=224,
            patch_size=16,
            embed_dim=256,
            depth=8
        )

        # From Step-4B
        self.encoder = KANGraphEncoder()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.30)

        # Classification Head
        self.cls_head = nn.Sequential(

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(128,num_classes)

        )

        # Regression Head
        self.reg_head = nn.Sequential(

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Linear(128,1)

        )

    def forward(self,x):

        x = self.backbone(x)

        x = self.encoder(x)

        x = x.transpose(1,2)

        x = self.pool(x).squeeze(-1)

        x = self.dropout(x)

        cls = self.cls_head(x)

        reg = self.reg_head(x)

        return cls,reg

# ----------------------------------------------------------
# Composite Loss
# ----------------------------------------------------------

class MKGLoss(nn.Module):

    def __init__(self,
                 alpha=1.0,
                 beta=0.5):

        super().__init__()

        self.alpha=alpha

        self.beta=beta

        self.ce=nn.CrossEntropyLoss()

        self.mse=nn.MSELoss()

    def forward(self,
                pred_cls,
                pred_reg,
                target_cls,
                target_reg):

        loss1=self.ce(
            pred_cls,
            target_cls
        )

        loss2=self.mse(
            pred_reg.squeeze(),
            target_reg.float()
        )

        loss=self.alpha*loss1+self.beta*loss2

        return loss

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

device=torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

model=MKGNet(
    num_classes=2
).to(device)

criterion=MKGLoss()

optimizer=torch.optim.AdamW(

    model.parameters(),

    lr=1e-4,

    weight_decay=1e-5

)

scheduler=torch.optim.lr_scheduler.StepLR(

    optimizer,

    step_size=10,

    gamma=0.5

)

print(model)

# ----------------------------------------------------------
# Train Function
# ----------------------------------------------------------

def train_one_epoch(loader):

    model.train()

    total_loss=0

    correct=0

    total=0

    for images,labels,lst in loader:

        images=images.to(device)

        labels=labels.to(device)

        lst=lst.to(device)

        optimizer.zero_grad()

        out_cls,out_reg=model(images)

        loss=criterion(

            out_cls,

            out_reg,

            labels,

            lst

        )

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

        pred=torch.argmax(out_cls,1)

        correct+=(pred==labels).sum().item()

        total+=labels.size(0)

    scheduler.step()

    return total_loss/len(loader),100*correct/total

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

@torch.no_grad()

def validate(loader):

    model.eval()

    total_loss=0

    correct=0

    total=0

    for images,labels,lst in loader:

        images=images.to(device)

        labels=labels.to(device)

        lst=lst.to(device)

        out_cls,out_reg=model(images)

        loss=criterion(

            out_cls,

            out_reg,

            labels,

            lst

        )

        total_loss+=loss.item()

        pred=torch.argmax(out_cls,1)

        correct+=(pred==labels).sum().item()

        total+=labels.size(0)

    return total_loss/len(loader),100*correct/total

# ----------------------------------------------------------
# Example Training
# ----------------------------------------------------------

EPOCHS=50

for epoch in range(EPOCHS):

    train_loss,train_acc=train_one_epoch(train_loader)

    val_loss,val_acc=validate(val_loader)

    print(

        f"Epoch {epoch+1:03d}"

        f" | Train Loss {train_loss:.4f}"

        f" | Train Acc {train_acc:.2f}"

        f" | Val Loss {val_loss:.4f}"

        f" | Val Acc {val_acc:.2f}"

    )

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

torch.save(

    model.state_dict(),

    "MKGNet_Model.pth"

)

print("\nMK-GNet Training Completed Successfully")

# ==========================================================
# STEP 5A : AGENT-RARL (RAG + CoVe)
# ==========================================================

# pip install langchain faiss-cpu sentence-transformers transformers

import os
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------
# Load Embedding Model
# ----------------------------------------------------------

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------------------------------------
# Load LLM
# ----------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)

llm = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# ----------------------------------------------------------
# Knowledge Base
# ----------------------------------------------------------

documents = [

"High NDVI indicates vegetation.",

"High NDBI represents built-up area.",

"High Land Surface Temperature indicates Urban Heat Island.",

"Green infrastructure reduces heat.",

"Dense buildings increase thermal stress.",

"Water bodies reduce daytime temperature.",

"Urban trees reduce LST.",

"Impervious surfaces increase UHI."

]

# ----------------------------------------------------------
# Create Embeddings
# ----------------------------------------------------------

doc_embeddings = embedder.encode(documents)

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(doc_embeddings))

# ----------------------------------------------------------
# Retrieval
# ----------------------------------------------------------

def retrieve(query, topk=3):

    q = embedder.encode([query])

    distance, ids = index.search(q, topk)

    result = []

    for i in ids[0]:

        result.append(documents[i])

    return result

# ----------------------------------------------------------
# Chain of Verification
# ----------------------------------------------------------

def verify(retrieved):

    verified=[]

    for sentence in retrieved:

        if len(sentence)>10:

            verified.append(sentence)

    return verified

# ----------------------------------------------------------
# Prompt Builder
# ----------------------------------------------------------

def build_prompt(query):

    docs=retrieve(query)

    docs=verify(docs)

    context="\n".join(docs)

    prompt=f"""

Context:

{context}

Question:

{query}

Answer:

"""

    return prompt

# ----------------------------------------------------------
# Generate Response
# ----------------------------------------------------------

def ask_agent(question):

    prompt=build_prompt(question)

    inputs=tokenizer(

        prompt,

        return_tensors="pt"

    ).to(llm.device)

    output=llm.generate(

        **inputs,

        max_new_tokens=150,

        do_sample=True,

        temperature=0.3

    )

    answer=tokenizer.decode(

        output[0],

        skip_special_tokens=True

    )

    return answer

# ----------------------------------------------------------
# Example
# ----------------------------------------------------------

question="Why is this region classified as high heat risk?"

reply=ask_agent(question)

print(reply)
# ==========================================================
# STEP 5B : PPO AGENT + ERI CALCULATION
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Policy Network
# ==========================================================

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim=256, action_dim=5):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(state_dim,256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,action_dim)

        )

    def forward(self,x):

        logits=self.model(x)

        return torch.softmax(logits,dim=-1)


# ==========================================================
# Value Network
# ==========================================================

class ValueNetwork(nn.Module):

    def __init__(self,state_dim=256):

        super().__init__()

        self.model=nn.Sequential(

            nn.Linear(state_dim,256),

            nn.ReLU(),

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Linear(128,1)

        )

    def forward(self,x):

        return self.model(x)


# ==========================================================
# PPO Agent
# ==========================================================

class PPOAgent:

    def __init__(self,
                 state_dim=256,
                 action_dim=5):

        self.policy=PolicyNetwork(
            state_dim,
            action_dim
        ).to(device)

        self.value=ValueNetwork(
            state_dim
        ).to(device)

        self.policy_optimizer=optim.Adam(
            self.policy.parameters(),
            lr=3e-4
        )

        self.value_optimizer=optim.Adam(
            self.value.parameters(),
            lr=1e-3
        )

        self.gamma=0.99

        self.clip=0.2

    def choose_action(self,state):

        probs=self.policy(state)

        dist=Categorical(probs)

        action=dist.sample()

        logprob=dist.log_prob(action)

        return action,logprob

    def compute_return(self,rewards):

        returns=[]

        R=0

        for reward in reversed(rewards):

            R=reward+self.gamma*R

            returns.insert(0,R)

        return torch.tensor(
            returns,
            dtype=torch.float32
        ).to(device)

    def update(self,
               states,
               actions,
               old_logprob,
               rewards):

        returns=self.compute_return(rewards)

        probs=self.policy(states)

        dist=Categorical(probs)

        new_logprob=dist.log_prob(actions)

        ratio=torch.exp(
            new_logprob-old_logprob
        )

        values=self.value(states).squeeze()

        advantage=returns-values.detach()

        surr1=ratio*advantage

        surr2=torch.clamp(

            ratio,

            1-self.clip,

            1+self.clip

        )*advantage

        policy_loss=-torch.min(
            surr1,
            surr2
        ).mean()

        value_loss=nn.MSELoss()(
            values,
            returns
        )

        self.policy_optimizer.zero_grad()

        policy_loss.backward()

        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()

        value_loss.backward()

        self.value_optimizer.step()

        return policy_loss.item(),value_loss.item()


# ==========================================================
# Environmental Risk Index
# ==========================================================

def calculate_eri(

        lst,

        ndvi,

        ndbi,

        population,

        humidity

):

    lst=(lst-lst.min())/(lst.max()-lst.min()+1e-8)

    ndvi=(ndvi-ndvi.min())/(ndvi.max()-ndvi.min()+1e-8)

    ndbi=(ndbi-ndbi.min())/(ndbi.max()-ndbi.min()+1e-8)

    population=(population-population.min())/(population.max()-population.min()+1e-8)

    humidity=(humidity-humidity.min())/(humidity.max()-humidity.min()+1e-8)

    eri=(

        0.35*lst+

        0.25*ndbi+

        0.20*population+

        0.10*(1-ndvi)+

        0.10*(1-humidity)

    )

    return eri


# ==========================================================
# Risk Category
# ==========================================================

def classify_risk(eri):

    risk=[]

    for value in eri:

        if value<0.20:

            risk.append("Very Low")

        elif value<0.40:

            risk.append("Low")

        elif value<0.60:

            risk.append("Moderate")

        elif value<0.80:

            risk.append("High")

        else:

            risk.append("Very High")

    return risk


# ==========================================================
# Example
# ==========================================================

agent=PPOAgent()

dummy_state=torch.randn(
    32,
    256
).to(device)

action,logprob=agent.choose_action(
    dummy_state
)

print("Action Shape :",action.shape)

print("LogProb Shape :",logprob.shape)

# Dummy Environmental Parameters

lst=np.random.rand(100)

ndvi=np.random.rand(100)

ndbi=np.random.rand(100)

population=np.random.rand(100)

humidity=np.random.rand(100)

eri=calculate_eri(

    lst,

    ndvi,

    ndbi,

    population,

    humidity

)

risk=classify_risk(eri)

print("\nFirst 10 ERI Values")

print(eri[:10])

print("\nFirst 10 Risk Labels")

print(risk[:10])

print("\nSTEP-5B COMPLETED SUCCESSFULLY")
# ==========================================================
# STEP 6 : MODEL TESTING & EVALUATION
# ==========================================================

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Load Trained Model
# ----------------------------------------------------------

model = MKGNet(num_classes=2)

model.load_state_dict(
    torch.load(
        "MKGNet_Model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model Loaded Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

all_pred=[]

all_true=[]

all_prob=[]

all_lst=[]

pred_lst=[]

with torch.no_grad():

    for images,labels,lst in test_loader:

        images=images.to(device)

        labels=labels.to(device)

        lst=lst.to(device)

        out_cls,out_reg=model(images)

        prob=torch.softmax(
            out_cls,
            dim=1
        )

        pred=torch.argmax(
            prob,
            dim=1
        )

        all_pred.extend(
            pred.cpu().numpy()
        )

        all_true.extend(
            labels.cpu().numpy()
        )

        all_prob.extend(
            prob[:,1].cpu().numpy()
        )

        pred_lst.extend(
            out_reg.squeeze().cpu().numpy()
        )

        all_lst.extend(
            lst.cpu().numpy()
        )

print("Prediction Completed")

# ----------------------------------------------------------
# Classification Metrics
# ----------------------------------------------------------

accuracy=accuracy_score(
    all_true,
    all_pred
)

precision=precision_score(
    all_true,
    all_pred
)

recall=recall_score(
    all_true,
    all_pred
)

f1=f1_score(
    all_true,
    all_pred
)

print("\n===========================")
print("Classification Result")
print("===========================")

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

# ----------------------------------------------------------
# Regression Metrics
# ----------------------------------------------------------

mae=np.mean(
    np.abs(
        np.array(pred_lst)-np.array(all_lst)
    )
)

rmse=np.sqrt(

    np.mean(

        (

            np.array(pred_lst)-np.array(all_lst)

        )**2

    )

)

print("\n===========================")
print("Regression Result")
print("===========================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report\n")

print(

classification_report(

all_true,

all_pred,

target_names=[

"Normal",

"High UHI"

]

)

)

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm=confusion_matrix(

all_true,

all_pred

)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Save Prediction
# ----------------------------------------------------------

results=pd.DataFrame({

"True_Label":all_true,

"Predicted_Label":all_pred,

"Probability":all_prob,

"True_LST":all_lst,

"Predicted_LST":pred_lst

})

results.to_csv(

"Prediction_Result.csv",

index=False

)

print("\nPrediction_Result.csv Saved Successfully")

print("\nSTEP-6 COMPLETED SUCCESSFULLY")
# ==========================================================
# STEP 7 : RESULT VISUALIZATION
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(all_true, all_pred)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Normal","High UHI"]

)

plt.figure(figsize=(6,6))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(

    "Confusion_Matrix.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

fpr,tpr,threshold = roc_curve(

    all_true,

    all_prob

)

roc_auc = auc(

    fpr,

    tpr

)

plt.figure(figsize=(7,6))

plt.plot(

    fpr,

    tpr,

    linewidth=3,

    label=f"AUC = {roc_auc:.4f}"

)

plt.plot(

    [0,1],

    [0,1],

    '--'

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "ROC_Curve.png",

    dpi=300

)

plt.show()

# ==========================================================
# Precision Recall Curve
# ==========================================================

precision,recall,_ = precision_recall_curve(

    all_true,

    all_prob

)

ap = average_precision_score(

    all_true,

    all_prob

)

plt.figure(figsize=(7,6))

plt.plot(

    recall,

    precision,

    linewidth=3,

    label=f"AP={ap:.4f}"

)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "PR_Curve.png",

    dpi=300

)

plt.show()

# ==========================================================
# Regression Scatter Plot
# ==========================================================

plt.figure(figsize=(7,7))

plt.scatter(

    all_lst,

    pred_lst,

    alpha=0.7

)

plt.plot(

    [min(all_lst),max(all_lst)],

    [min(all_lst),max(all_lst)],

    'r--'

)

plt.xlabel("Actual LST")

plt.ylabel("Predicted LST")

plt.title("Actual vs Predicted LST")

plt.grid(True)

plt.savefig(

    "Regression.png",

    dpi=300

)

plt.show()

# ==========================================================
# Prediction Error
# ==========================================================

error = np.array(pred_lst)-np.array(all_lst)

plt.figure(figsize=(8,5))

plt.hist(

    error,

    bins=30

)

plt.title("Prediction Error Distribution")

plt.xlabel("Error")

plt.ylabel("Frequency")

plt.grid(True)

plt.savefig(

    "Prediction_Error.png",

    dpi=300

)

plt.show()

# ==========================================================
# ERI Distribution
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(

    eri,

    bins=25

)

plt.title("Environmental Risk Index")

plt.xlabel("ERI")

plt.ylabel("Frequency")

plt.grid(True)

plt.savefig(

    "ERI_Distribution.png",

    dpi=300

)

plt.show()

print("\nGraphs Saved Successfully")

print("""
Saved Figures

1. Confusion_Matrix.png
2. ROC_Curve.png
3. PR_Curve.png
4. Regression.png
5. Prediction_Error.png
6. ERI_Distribution.png
""")

print("STEP-7 COMPLETED SUCCESSFULLY")