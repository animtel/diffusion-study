# diffusion-study
Custom implementation of iterative image diffusion process

### Build & Run docker

- `docker build -t sdiffusion .` - to build docker
- To run a container, use:
```
docker run -d -it --init \
--gpus=all \
--ipc=host \
--volume="$PWD:/app" \
--volume="/home/:/hhome" \
--volume="/usr/local/cuda:/usr/local/cuda" \
--publish="5555:5555" \
--publish="5556:5556" \
sdiffusion bash
```

#### Run jupyter server

- `jupyter lab --no-browser --ip 0.0.0.0 --port 5555 --allow-root`

### Useful materials:

- [Yannic Kilcher | DDPM - Diffusion Models Beat GANs on Image Synthesis (Machine Learning Research Paper Explained)](https://youtu.be/W-O7AZNzbzQ)
- [Jeremy Howard | Lesson 9: Deep Learning Foundations to Stable Diffusion, 2022)](https://youtu.be/_7rMfsA24Ls)
- https://github.com/fastai/diffusion-nbs - Tools and Notebooks from fast.ai
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Tools and Resources for AI Art](https://pharmapsychotic.com/tools.html)
- [Private Local Waifu Diffusion colab](https://rentry.org/nocrypt)
- [Stable Diffusion Tutorial Part 1: Run Dreambooth in Gradient Notebooks](https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-1/)
- [Stable Diffusion Tutorial Part 2: Using Textual Inversion Embeddings to gain substantial control over your generated images](https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-part-2-textual-inversion/)
- [Generating images with Stable Diffusion](https://blog.paperspace.com/generating-images-with-stable-diffusion/)
- [Stable Diffusion Models](https://rentry.org/sdmodels#stable-diffusion-models)
- [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | [GitHub](https://github.com/CompVis/latent-diffusion)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [DiffRF: Rendering-guided 3D Radiance Field Diffusion](https://sirwyver.github.io/DiffRF/)
- [3D Neural Field Generation using Triplane Diffusion](https://jryanshue.com/nfd/)
- [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class)
- [Diffusion samplers easy to use implementations](https://github.com/ozanciga/diffusion-for-beginners)

---
### Datasets

- DDPM(Denoising Diffusion Probabilistic Models) was trained on [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
