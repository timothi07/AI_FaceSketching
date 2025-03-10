from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.base import ContentFile
from .models import GeneratedImage
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from django.conf import settings
from .inference_gfpgan import enhance_image_with_gfpgan
from django.urls import reverse

# Define the SentenceEncoder class
class SentenceEncoder:
    def __init__(self, device):
        self.bert_model = SentenceTransformer("all-mpnet-base-v2").to(device)
        self.device = device

    def convert_text_to_embeddings(self, batch_text):
        stack = []
        for sent in batch_text:
            sentences = sent.split(". ")
            sentence_embeddings = self.bert_model.encode(sentences)
            sentence_emb = torch.FloatTensor(sentence_embeddings).to(self.device)
            sent_mean = torch.mean(sentence_emb, dim=0).reshape(1, -1)
            stack.append(sent_mean)
        output = torch.cat(stack, dim=0)
        return output.detach()

# Define the Generator class
class Generator(nn.Module):
    '''
    The Generator Network
    '''
    def __init__(self, noise_size, feature_size, num_channels, embedding_size, reduced_dim_size):
        super(Generator, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=reduced_dim_size),
            nn.BatchNorm1d(num_features=reduced_dim_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(noise_size + reduced_dim_size, feature_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size * 4, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, text_embeddings):
        encoded_text = self.projection(text_embeddings)
        concat_input = torch.cat([noise, encoded_text], dim=1).unsqueeze(2).unsqueeze(2)
        output = self.layer(concat_input)
        return output

# Initialize SentenceEncoder and Generator
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sentence_encoder = SentenceEncoder(device)
generator = Generator(100, 128, 3, 768, 256).to(device)
#generator.load_state_dict(torch.load('Downloads/trial/generator.pth', map_location=device))

generator.load_state_dict(torch.load(os.path.join(settings.MEDIA_ROOT, 'generator.pth'), map_location=device))
generator.eval()

def index(request):
    return render(request, 'generator/index.html')

# def result(request, image_id):
#     return render(request, "generator/result.html", {"image_id": image_id})

# def result(request, image_id):
#     image = get_object_or_404(GeneratedImage, id=image_id)
#     return render(request, 'generator/result.html', {'image': image})
def result(request, image_id):
    image = get_object_or_404(GeneratedImage, id=image_id)  # Ensure the image is retrieved correctly
    return render(request, 'generator/result.html', {'image': image, 'image_id': image.id})


def generate_image(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        test_noise = torch.randn(1, 100, device=device)
        test_embeddings = sentence_encoder.convert_text_to_embeddings([input_text])
        
        with torch.no_grad():
            test_image = generator(test_noise, test_embeddings).detach().cpu()[0]
        
        np_image = test_image.permute(1, 2, 0).numpy()
        np_image = (np_image + 1) / 2
        pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
        
        # Save the generated image
        image_path = 'generated_images/generated_image.png'
        pil_image.save(image_path)
        
        # Save to database (optional)
        generated_image = GeneratedImage(description=input_text)
        generated_image.image.save('generated_image.png', ContentFile(open(image_path, 'rb').read()))
        generated_image.save()
        
        #return render(request, 'generator/generate.html', {'generated_image': generated_image})
        return redirect('generator:result', image_id=generated_image.id)
    
    #return render(request, 'generator/generate.html')
    return redirect('index')

def enhance_image(request, image_id):
    #generated_image = GeneratedImage.objects.get(id=image_id)
    generated_image = get_object_or_404(GeneratedImage, id=image_id)
    
    # Path to the generated image
    input_image_path = generated_image.image.path
    
    # Output directory for enhanced images
    output_dir = 'enhanced_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhance the image using GFPGAN
    enhanced_image_path = enhance_image_with_gfpgan(
        input_path=input_image_path,
        output_path=output_dir,
        version='1.3',  # GFPGAN version
        upscale=2,  # Upscaling factor
        bg_upsampler='realesrgan',  # Background upsampler
        weight=0.5  # Adjustable weights
    )
    
    # Save the enhanced image to the database
    generated_image.enhanced_image.save(os.path.basename(enhanced_image_path), ContentFile(open(enhanced_image_path, 'rb').read()))
    generated_image.save()
    
    #return render(request, 'generator/generate.html', {'generated_image': generated_image})
    return redirect('generator:result', image_id=generated_image.id)

    



