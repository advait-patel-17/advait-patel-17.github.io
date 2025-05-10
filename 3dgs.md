Ignore this. It's a blog post that I had an LLM write for a class to see how well it could do.

# Beyond the Hype: Understanding 3D Gaussian Splatting

*A Deep Dive into 2023's Real-Time Rendering Sensation*

Estimated reading time: 12-15 minutes

Remember those sci-fi movies where someone waves a scanner and a perfect 3D hologram of a room appears instantly? For years, that felt like pure fiction. Rendering detailed 3D scenes in real-time, especially from real-world photos, was a slow, arduous process. Then, in 2023, a paper landed like a meteor: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al.). Suddenly, high-fidelity, real-time rendering from images wasn't just possible; it was happening, and it looked *stunning*. It felt like the graphics world collectively gasped, then scrambled to understand how these "fuzzy paintballs" were achieving such magic. This post unpacks the what, why, and how of 3D Gaussian Splatting.

## Table of Contents

*   [1. The Intuition: Programmable 3D Paintballs](#1-the-intuition-programmable-3d-paintballs)
*   [2. The Building Blocks: What is a 3D Gaussian?](#2-the-building-blocks-what-is-a-3d-gaussian)
*   [3. The Magic Recipe: The Gaussian Splatting Pipeline](#3-the-magic-recipe-the-gaussian-splatting-pipeline)
    *   [Step 1: Starting with Points - Structure-from-Motion (SfM)](#step-1-starting-with-points---structure-from-motion-sfm)
    *   [Step 2: Initializing Our Gaussians](#step-2-initializing-our-gaussians)
    *   [Step 3: Differentiable Rasterization - The Secret Sauce](#step-3-differentiable-rasterization---the-secret-sauce)
    *   [Step 4: Learning and Refining - Optimization and Adaptive Density Control](#step-4-learning-and-refining---optimization-and-adaptive-density-control)
*   [4. A Glimpse Under the Hood: Minimal Rasterizer Code](#4-a-glimpse-under-the-hood-minimal-rasterizer-code)
*   [5. How Does It Stack Up? Gaussian Splatting vs. The World](#5-how-does-it-stack-up-gaussian-splatting-vs-the-world)
*   [6. The Road Ahead: Applications and Future Directions](#6-the-road-ahead-applications-and-future-directions)
*   [Hands-On with Gaussian Splatting](#hands-on-with-gaussian-splatting)
*   [References](#references)

## 1. The Intuition: Programmable 3D Paintballs

Imagine you want to recreate a complex 3D scene, like a cluttered room or a detailed statue. Instead of meticulously building a geometric mesh (like in traditional CGI) or training a complex neural network to predict color and density along rays (like NeRFs), what if you could just throw millions of tiny, semi-transparent, colored "paintballs" at it?

This is the core idea behind 3D Gaussian Splatting. Each "paintball" is a 3D Gaussian – a mathematical entity that describes a fuzzy blob in space. These aren't just simple spheres; they can be stretched, squashed, and rotated, allowing them to represent all sorts of shapes and surfaces. Each Gaussian also has a color and an opacity (how see-through it is).

When you want to view the scene from a particular angle, you "splat" these 3D Gaussians onto your 2D screen, much like throwing paintballs at a canvas. The clever part is how these splats are combined: they are blended together based on their depth, color, and opacity to form a final, cohesive image. And because this whole process is "differentiable" (more on that later), we can use machine learning techniques to automatically adjust the position, shape, color, and opacity of millions of these Gaussians until they perfectly reconstruct the scene from a set of input photos. The result? Photorealistic renderings at lightning-fast speeds, often exceeding 100 frames per second (Kerbl et al.).

As Jonathan Stephens aptly put it, Gaussian Splatting marked "a new era for 3D," combining the explicit nature of point clouds with the rendering quality approaching that of neural radiance fields, but at significantly faster speeds (Stephens).

---
**Key Takeaway:** 3D Gaussian Splatting represents a scene as millions of tiny, optimizable 3D Gaussian "blobs," which are then "splatted" onto the screen for fast, high-quality rendering.
---

## 2. The Building Blocks: What is a 3D Gaussian?

To understand the "splatting," we first need to understand the "Gaussian." A 3D Gaussian function describes a probability distribution in 3D space. In our context, it's a blob whose density falls off smoothly from its center.

Each 3D Gaussian in the scene is defined by several key parameters:

---
**Notation Panel**

*   $\mu$ (mu): Mean, a 3D vector $(x, y, z)$ representing the **position** or center of the Gaussian.
*   $\Sigma$ (Sigma): Covariance matrix, a $3 \times 3$ matrix defining the **shape and orientation** of the Gaussian. Think of it as controlling how the Gaussian is stretched, squashed, and rotated.
    *   In practice, $\Sigma$ is often parameterized by a 3D scaling vector $S = (s_x, s_y, s_z)$ and a quaternion $R$ (representing rotation) for easier optimization. $\Sigma = R S S^T R^T$.
*   $c$: Color, typically an RGB triplet $(r, g, b)$. The original paper uses Spherical Harmonics (SH) coefficients to represent view-dependent color, but for simplicity, we can think of it as RGB.
*   $\alpha$ (alpha): Opacity, a scalar value between 0 (fully transparent) and 1 (fully opaque).

---

The unnormalized Gaussian function $G(x)$ for a point $x$ relative to a Gaussian centered at $\mu$ with covariance $\Sigma$ is given by:

$G(x) = e^{-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)}$

This formula tells us how "intense" the Gaussian is at any given point $x$ in 3D space. The further $x$ is from the center $\mu$, or the more it deviates along directions where the covariance $\Sigma$ indicates the Gaussian is "thin," the smaller $G(x)$ becomes. When we render, we multiply this intensity by the Gaussian's learned color $c$ and opacity $\alpha$.

The ability to optimize these parameters ($\mu, \Sigma, c, \alpha$) for millions of Gaussians simultaneously is what gives the method its power.

*Diagram Idea: A 3D ellipsoid representing a Gaussian.*
content_copy
download
Use code with caution.
Markdown
z
  |
 / \
|---|(Sigma_y)
content_copy
download
Use code with caution.
/
/-------\ (Sigma_x)
/
o-------------> x (mu)
\ /
-------/
-----/ (Sigma_z projected)
\ /
\ /
y (projected)

Alt: A 3D ellipsoid centered at mu, illustrating its extent along x, y, and z axes, determined by its covariance matrix Sigma. This represents a single 3D Gaussian.

---
**Key Takeaway:** A 3D Gaussian is a mathematical primitive defined by its position (mean), shape/orientation (covariance), color, and opacity, forming the fundamental building block of the rendered scene.
---

## 3. The Magic Recipe: The Gaussian Splatting Pipeline

The journey from a collection of 2D images to a real-time 3D rendered scene involves a clever multi-stage pipeline. The original authors laid out a system that is both elegant and effective (Kerbl et al.).

*Diagram Idea: Overall Pipeline Flowchart.*
content_copy
download
Use code with caution.
[Input Images] --> [Step 1: SfM (COLMAP)] --> [Sparse Point Cloud & Camera Poses]
|
v
[Step 2: Initialize 3D Gaussians]
|
v
------------------------------------
| Training Loop |
| |
| [Step 3: Differentiable |
| Rasterization] | ----> [Rendered Image]
| (Project & Blend) | |
| | | (Compare)
| [Step 4: Optimization & | |
| Adaptive Density | <--- [Ground Truth Image]
| Control (Densify/Prune)] (Loss)
------------------------------------
|
v
[Optimized 3D Gaussians] --> [Real-Time Rendering]

Alt: Flowchart of the 3D Gaussian Splatting pipeline. Input images feed into Structure-from-Motion (SfM). SfM outputs sparse points and camera poses, which initialize 3D Gaussians. These Gaussians enter a training loop: Render Image via Differentiable Rasterization, Compute Loss (compare with ground truth), Optimize Parameters, and Adaptive Density Control (densify/prune). The loop outputs optimized 3D Gaussians for real-time rendering.

Let's break down these steps:

### Step 1: Starting with Points - Structure-from-Motion (SfM)

The process begins with a set of images of a scene taken from various viewpoints.
1.  **Input:** A collection of 2D images.
2.  **Process:** A standard Structure-from-Motion (SfM) algorithm, like the popular COLMAP (Schoenberger and Frahm), is used. SfM analyzes the input images to simultaneously estimate:
    *   The 3D positions of a sparse set of points in the scene.
    *   The camera parameters (position, orientation, focal length) for each input image.
3.  **Output:** A sparse point cloud (XYZ coordinates for a few thousand to a few hundred thousand points) and calibrated camera poses.

This initial point cloud provides the scaffold upon which the Gaussians will be built.

### Step 2: Initializing Our Gaussians

The sparse points from SfM are converted into an initial set of 3D Gaussians. For each point in the SfM output:
*   **Position ($\mu$):** Set directly from the 3D coordinates of the SfM point.
*   **Covariance ($\Sigma$):** Initialized to be anisotropic, estimated from the distances to its nearest neighbors in the SfM point cloud. This helps the initial Gaussians roughly match the local geometry. For example, if neighbors are spread out along a plane, the Gaussian will be flatter.
*   **Color ($c$):** The original paper initializes color using Spherical Harmonics (SH) coefficients. A simpler approach is to set it to the color of the point as observed in the input images. The paper sets the 0th-order SH coefficient (diffuse color) from the SfM point's color.
*   **Opacity ($\alpha$):** Initialized to a small, positive value (e.g., 0.1) to make them mostly transparent at first.

This gives us a starting set of "raw" Gaussians, ready for optimization.

### Step 3: Differentiable Rasterization - The Secret Sauce

This is where the real innovation of 3D Gaussian Splatting shines. To optimize the Gaussians, we need to render them and compare the rendering to the original input images. This rendering process must be differentiable, meaning we can calculate how a small change in any Gaussian parameter ($\mu, \Sigma, c, \alpha$) affects the final rendered pixel colors.

For each training view (one of the input images):
1.  **Projection:** Each 3D Gaussian is projected onto the 2D image plane of the current camera view. This involves:
    *   Transforming its 3D mean $\mu$ to a 2D mean $\mu_{2D}$ on the image.
    *   Transforming its 3D covariance $\Sigma$ into a 2D covariance matrix $\Sigma_{2D}$ that defines the elliptical shape of the "splat" on the image. This projection is non-trivial and involves the Jacobian of the affine projection matrix and the 3D covariance: $\Sigma_{2D} = J \Sigma J^T$.

    *Diagram Idea: Gaussian Projection.*
    ```
    3D Space                                  2D Image Plane
    ---------                                 --------------
       Mu_3D (center)
       /|\
      / | \ Covariance_3D (ellipsoid)
     -----
      \|/
       V (View Direction)
    Projection (Camera Transformation)
       |
       V
                                        Mu_2D (center)
                                       //===\\ Covariance_2D (ellipse)
                                       ||     || "Splat"
                                       \\===//

    Alt: Diagram showing a 3D Gaussian (an ellipsoid in 3D space) being projected by a camera onto a 2D image plane. The result is a 2D ellipse (a splat) on the image.
    ```

2.  **Sorting:** The Gaussians are sorted by their depth (distance from the camera) in a front-to-back order. This is crucial for correct alpha blending.

    > **Pro-Tip: Fast Sorting is Key**
    > Sorting millions of Gaussians per frame can be a bottleneck. The original paper implements a highly optimized, GPU-accelerated radix sort, achieving sorting in milliseconds. This is a significant engineering feat contributing to the real-time performance.

3.  **Blending (Alpha Compositing):** For each pixel on the screen, the colors and opacities of all 2D Gaussian splats that cover that pixel are blended together. The color $C$ of a pixel is accumulated by:
    $C = \sum_{i=1}^{N} c_i \alpha_i' \prod_{j=1}^{i-1} (1 - \alpha_j')$
    where $N$ is the number of Gaussians overlapping the pixel (sorted by depth), $c_i$ is the color of the $i$-th Gaussian, and $\alpha_i'$ is its projected opacity at the pixel center (derived from $\alpha$ and the 2D Gaussian function). Each Gaussian contributes its color, attenuated by its own opacity and the accumulated opacity of all Gaussians in front of it.

    *Diagram Idea: Alpha Blending.*
    ```
    Ray from Camera -->
                    Pixel
                      |
    Splat 1 (front) c1, a1  -> Contrib: c1 * a1
                      |
    Splat 2         c2, a2  -> Contrib: c2 * a2 * (1 - a1)
                      |
    Splat 3 (back)  c3, a3  -> Contrib: c3 * a3 * (1 - a1) * (1 - a2)
                      |
                      V
                   Final Pixel Color = Sum of Contributions

    Alt: Schematic showing multiple semi-transparent 2D splats overlapping along a ray. Splat 1 (closest) contributes its color multiplied by its alpha. Splat 2's contribution is its color multiplied by its alpha and (1 - alpha of Splat 1). Splat 3's contribution is further attenuated by the opacities of Splat 1 and Splat 2.
    ```

The entire rasterization pipeline, from projection to blending, is designed to be differentiable. This means that if the rendered image doesn't match the ground truth, we can compute gradients that tell us how to adjust each Gaussian's $\mu, \Sigma, c, \alpha$ to reduce this error.

### Step 4: Learning and Refining - Optimization and Adaptive Density Control

With a differentiable rasterizer, we can now train our Gaussians:
1.  **Loss Calculation:** The rendered image is compared to the corresponding ground truth input image. The loss function is typically a combination of an L1 loss and a D-SSIM (Structural Dissimilarity Index Measure) loss, which captures both per-pixel differences and perceptual similarity.
2.  **Optimization:** Gradients of the loss with respect to all Gaussian parameters ($\mu, \Sigma, c, \alpha$) are computed via automatic differentiation (e.g., PyTorch's autograd). These gradients are then used by an optimizer (like Adam) to update the parameters.
3.  **Adaptive Density Control:** This is a crucial step that happens periodically during training (e.g., every 100 iterations). The initial SfM points are sparse, so we need to intelligently add more Gaussians where needed and remove useless ones.
    *   **Densification:**
        *   **Cloning for Undersampling:** If a Gaussian is in an area with high reconstruction error (large positional gradients), it's cloned, and the clone is moved slightly in the direction of this gradient.
        *   **Splitting for Oversampling:** If a Gaussian covers a large area in view-space (meaning it's too big and trying to represent too much detail), it's split into two smaller Gaussians.
    *   **Pruning:** Gaussians whose opacity $\alpha$ drops below a certain threshold (they become almost invisible) are removed. Gaussians that are extremely small might also be pruned.
    *   The opacity of Gaussians is also reset to near zero periodically to prevent them from becoming fixed too early.

    > **Pitfall: VRAM Bloat**
    > Adaptive densification is powerful, but if not controlled carefully, it can lead to an explosion in the number of Gaussians. This consumes vast amounts of VRAM and can slow down both training and rendering. The pruning strategy and careful thresholds for densification are vital to keep the model manageable while achieving high quality.

This iterative process of rendering, loss calculation, optimization, and adaptive density control continues for thousands of iterations until the rendered images closely match the training images from all viewpoints.

---
**Key Takeaway:** The Gaussian Splatting pipeline initializes Gaussians from SfM points, then iteratively refines them through a differentiable rasterization process, coupled with an adaptive mechanism to add or remove Gaussians where needed.
---

## 4. A Glimpse Under the Hood: Minimal Rasterizer Code

The full CUDA rasterizer from the paper is highly complex and optimized. However, we can illustrate the core idea of projecting and blending with a conceptual PyTorch snippet. This is heavily simplified and focuses on a single Gaussian for clarity; a real implementation handles millions.

```python
import torch
import torch.nn.functional as F

# Assume we have one Gaussian's parameters (already optimized or being optimized)
# For simplicity, we'll use 2D parameters directly here.
# A full implementation would project 3D Gaussians to 2D.

# Gaussian parameters (requires_grad=True for optimization)
mean2D = torch.tensor([128.0, 128.0], requires_grad=True) # Center in pixels
# Simplified covariance: scaling factors and a rotation angle
scales = torch.tensor([30.0, 15.0], requires_grad=True) # Ellipse radii
angle_rad = torch.tensor([0.785], requires_grad=True) # 45 degrees
color_rgba = torch.tensor([0.2, 0.8, 0.3, 0.9], requires_grad=True) # RGBA

# Image canvas
H, W = 256, 256
image_canvas = torch.zeros((H, W, 4)) # RGBA image

# Create a grid of pixel coordinates
grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
coords = torch.stack((grid_x, grid_y), dim=-1).float() # Shape (H, W, 2)

# --- Simplified Projection & Splatting of ONE Gaussian ---
# 1. Transform coordinates to Gaussian's local frame
rot_matrix = torch.tensor([
    [torch.cos(angle_rad), -torch.sin(angle_rad)],
    [torch.sin(angle_rad),  torch.cos(angle_rad)]
])
# Inverse operations for transforming grid to Gaussian space
# (Translate grid so Gaussian mean is origin, then rotate, then scale)
translated_coords = coords - mean2D
rotated_coords = torch.matmul(translated_coords.view(-1, 2), rot_matrix.T).view(H, W, 2)
# Scaled distance (Mahalanobis distance squared, simplified)
# (x/sx)^2 + (y/sy)^2
dist_sq = (rotated_coords[..., 0] / scales[0])**2 + \
          (rotated_coords[..., 1] / scales[1])**2

# 2. Calculate Gaussian opacity falloff for each pixel
# opacity_at_pixel = base_opacity * exp(-0.5 * dist_sq)
# Ensure scales are positive during optimization via torch.exp or softplus
effective_scales = torch.exp(scales * 0.1) # ensure positive, dampen sensitivity
gaussian_alpha_map = color_rgba[3] * torch.exp(-0.5 * dist_sq) # Shape (H, W)

# 3. Color contribution for this Gaussian
# C_gaussian = color_rgb * gaussian_alpha_map
gaussian_color_contribution = color_rgba[:3] * gaussian_alpha_map.unsqueeze(-1) # Shape (H, W, 3)

# --- Simplified Alpha Blending (for one Gaussian, just assign) ---
# For multiple Gaussians, this would be an iterative process:
# current_alpha_acc = current_alpha_acc + (1 - current_alpha_acc) * gaussian_alpha_map
# image_canvas_rgb = image_canvas_rgb + (1 - current_alpha_acc_prev) * gaussian_color_contribution
# Here, we just form the RGBA splat for this single Gaussian
splat = torch.cat((gaussian_color_contribution, gaussian_alpha_map.unsqueeze(-1)), dim=-1)

# This 'splat' could then be used in a loss function
# e.g., loss = ((splat - target_splat)**2).mean()
# loss.backward() would populate grads for mean2D, scales, angle_rad, color_rgba

# For visualization:
# image_canvas = splat # (This is for one Gaussian)
# To display, you'd convert to a viewable format.

print("Conceptual splat calculation complete. Gradients can be computed for parameters.")
# In a real system, this happens for millions of Gaussians, sorted, and alpha-blended.
content_copy
download
Use code with caution.
This snippet demonstrates that operations like coordinate transformation, Gaussian evaluation, and color application can be expressed using PyTorch operations, making them automatically differentiable. The actual rasterizer would perform these steps for many Gaussians, project them from 3D to 2D, sort them by depth, and then carefully blend them onto a tile-based grid for efficiency. The Hugging Face blog on Gaussian Splatting offers a more detailed, yet still accessible, walkthrough of some of these concepts (Kunder).

Key Takeaway: The rasterization process, though complex, can be built from differentiable mathematical operations, allowing gradient-based optimization of all Gaussian parameters.
5. How Does It Stack Up? Gaussian Splatting vs. The World
3D Gaussian Splatting didn't emerge in a vacuum. It offers a compelling alternative to existing scene representation and rendering techniques. Here's a brief comparison:

Feature	3D Gaussian Splatting	Neural Radiance Fields (NeRF)	Meshes (Traditional CG)	Point Clouds (Raw)
Rendering Speed	Very Fast (Real-time, >100 FPS)	Slow to Moderate (seconds to ~10-30 FPS for fast variants)	Very Fast	Fast, but hard to shade well
Training Speed	Fast (minutes to <1 hour)	Slow (hours to days)	N/A (Manual Creation)	N/A (Directly Captured)
Visual Quality	Excellent, Photorealistic	Excellent, Photorealistic	Variable, artist-dependent	Sparse, lacks surface detail
Memory Footprint	Moderate to High (MBs to GBs for Gaussians)	Low to Moderate (Network weights)	Low to High	Moderate to High
Editability	Difficult (indirect control of Gaussians)	Very Difficult (implicit representation)	Excellent	Moderate (point editing)
View Dependence	Handled (via Spherical Harmonics)	Excellent (core NeRF feature)	Typically via shaders/textures	Poor
Dynamic Scenes	Emerging research (e.g., Dynamic 3D Gaussians)	Emerging research (e.g., D-NeRF, NeRFPlayer)	Complex, rigging/animation	Difficult to track
Requires Posed Images for Training?	Yes (often from SfM)	Yes	N/A	Can be used with SfM
NeRFs (Mildenhall et al.) were revolutionary for view synthesis quality, but their reliance on querying a neural network for every pixel along every ray makes them computationally intensive for both training and rendering. Gaussian Splatting, being an explicit representation, bypasses much of this per-pixel network evaluation during rendering.

Meshes are the workhorse of CGI and games. They are highly editable and efficient to render but typically require significant manual artistry or complex reconstruction algorithms to create from real-world data.

Point Clouds are a direct, raw representation of 3D data. While fast to render as points, achieving surface-like appearance and handling occlusions correctly without converting to another representation is challenging.

Gaussian Splatting strikes a remarkable balance: the training speed of traditional photogrammetry combined with rendering speeds suitable for real-time applications, and visual quality that rivals NeRFs.

Key Takeaway: Gaussian Splatting offers a compelling trade-off, achieving real-time rendering speeds and fast training times with photorealistic quality, outperforming NeRFs in speed and raw point clouds in quality and coherence.
6. The Road Ahead: Applications and Future Directions
The impact of 3D Gaussian Splatting was immediate and widespread, with researchers and industry quickly exploring its potential.

Current and Emerging Applications:

Real-Time View Synthesis & Walkthroughs: The most obvious application. Creating immersive digital replicas of real-world spaces that can be explored smoothly.
AR/VR Mapping and Experiences: Companies like Niantic are already using Gaussian Splatting to bring detailed, real-world scanned assets into augmented reality experiences with impressive fidelity. Their work on the character "Doty" showcases this power (Niantic Engineering).
Volumetric Video & Digital Avatars: Capturing and replaying dynamic performances or creating realistic digital humans. Meta has discussed evolving their Codec Avatars using Gaussian splats for hyper-realistic representations (Llamas et al.).
Robotics and Simulation: Generating realistic sensor data for training robots or simulating complex environments.
Game Development: Faster creation of game assets from real-world scans or conceptual art.
Cultural Heritage Preservation: Digitizing artifacts and sites with high fidelity for virtual tourism or archival.
Future Research Directions:

Dynamic Scenes: The original paper focused on static scenes. A major area of ongoing research is extending Gaussian Splatting to handle dynamic elements, deformable objects, and even human motion (e.g., Chen et al. with HumanGaussian).
Editing and Control: While powerful for reconstruction, direct, intuitive editing of Gaussian Splat scenes (e.g., "remove this car" or "change the texture of this wall") is still a challenge. Research is ongoing to provide better artistic control.
Compression and Streaming: Reducing the storage footprint of the millions of Gaussians and enabling efficient streaming for web or mobile applications.
Relighting and Material Editing: More sophisticated material models within the Gaussians to allow for realistic relighting or changing material properties.
Hybrid Representations: Combining Gaussian Splatting with other representations (e.g., meshes for coarse geometry, splats for fine detail) to get the best of multiple worlds.
The speed and quality offered by 3D Gaussian Splatting have undoubtedly unlocked new possibilities. As Keiran Crane from Two Minute Papers enthusiastically noted, "Gaussian Splatting is Awesome," capturing the community's excitement for its capabilities (Crane).

Key Takeaway: Gaussian Splatting is rapidly being adopted for AR/VR, volumetric video, and digital twins, with active research focused on dynamic scenes, editability, and compression.
Hands-On with Gaussian Splatting
Want to try it yourself? Here are some great starting points:

Original Paper & CUDA Implementation:
Project Page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
GitHub (CUDA Code): https://github.com/graphdeco-inria/gaussian-splatting
Popular PyTorch Implementation & Tools (easier to get started):
Nerfstudio: This toolkit has integrated Gaussian Splatting, making it accessible for training and experimentation.
GitHub: https://github.com/nerfstudio-project/nerfstudio
Gaussian Splatting Docs: https://docs.nerf.studio/en/latest/methodology/models/gaussian_splatting.html
Polycam / Luma AI: These services offer easy ways to capture data with your phone and process it into Gaussian Splats.
Colab Demos: Search for "Gaussian Splatting Colab" on Google. Several community-provided notebooks allow you to try training on sample datasets or your own images directly in your browser. For example, the Nerfstudio team often provides Colab links.
3D Gaussian Splatting has fundamentally changed the landscape of 3D scene reconstruction and real-time rendering. It's a brilliant combination of explicit representation, differentiable rendering, and adaptive optimization. While it has its own challenges and areas for improvement, its arrival marks a significant leap forward, bringing us one step closer to those sci-fi holograms we've always dreamed of.

References
Chen, Ang, et al. "HumanGaussian: Text-Driven Controllable Human Image Generation with 3D Gaussians." arXiv preprint arXiv:2311.17061, 2023.

Crane, Keiran. "Gaussian Splatting is Awesome." YouTube, uploaded by Two Minute Papers, 1 Aug. 2023, www.youtube.com/watch?v=jO6z9q22M5E.

Kerbl, Bernhard, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM Transactions on Graphics (TOG), vol. 42, no. 4, 2023, pp. 1-14. SIGGRAPH 2023.

Kunder, A. B. "Gaussian Splatting Explained." Hugging Face Blog, 24 Aug. 2023, huggingface.co/blog/gaussian-splatting.

Llamas, Jose Luis, et al. "Meta Avatars: Evolving our hyper-realistic Codec Avatars with Gaussian splats." Meta Tech Blog, 13 Mar. 2024, tech.meta.com/research/publications/meta-avatars-evolving-our-hyper-realistic-codec-avatars-with-gaussian-splats/.

Mildenhall, Ben, et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." European Conference on Computer Vision (ECCV), 2020.

Niantic Engineering. "Bringing Doty to Life with 3D Gaussian Splatting." Niantic Engineering Blog, 13 Dec. 2023, nianticlabs.com/news/gaussian-splatting.

Schoenberger, Johannes L., and Jan-Michael Frahm. "Structure-from-Motion Revisited." Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Stephens, Jonathan. "Nerfstudio and 3D Gaussian Splatting: A new era for 3D." Volumetrics.com Blog, 18 Oct. 2023, volumetrics.com/blog/nerfstudio-and-3d-gaussian-splatting-a-new-era-for-3d.

Weng, Lilian. "NeRF: Neural Radiance Field in JAX (Part 1: Model and Rendering)." Lil'Log, 12 July 2020, lilianweng.github.io/posts/2020-07-12-nerf/.

BibTeX Format:

@article{Kerbl2023GaussianSplatting,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"{u}}hler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics (TOG)},
  volume    = {42},
  number    = {4},
  year      = {2023},
  publisher = {ACM},
  note      = {SIGGRAPH 2023 Conference Proceedings}
}

@article{Mildenhall2020NeRF,
  author    = {Mildenhall, Ben and Srinivasan, Pratul P. and Tancik, Matthew and Barron, Jonathan T. and Ramamoorthi, Ravi and Ng, Ren},
  title     = {NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  journal   = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}

@online{Niantic2023DotySplatting,
  author    = {{Niantic Engineering}},
  title     = {Bringing Doty to Life with 3D Gaussian Splatting},
  year      = {2023},
  month     = {Dec},
  url       = {https://nianticlabs.com/news/gaussian-splatting}
}

@online{Stephens2023SplattingEra,
  author    = {Stephens, Jonathan},
  title     = {Nerfstudio and 3D Gaussian Splatting: A new era for 3D},
  year      = {2023},
  month     = {Oct},
  url       = {https://volumetrics.com/blog/nerfstudio-and-3d-gaussian-splatting-a-new-era-for-3d}
}

@online{Crane2023SplattingAwesome,
    author = {Crane, Keiran and {Two Minute Papers}},
    title = {Gaussian Splatting is Awesome},
    year = {2023},
    month = {Aug},
    url = {https://www.youtube.com/watch?v=jO6z9q22M5E},
    note = {YouTube Video}
}

@online{Meta2024AvatarsSplatting,
    author = {Llamas, Jose Luis and Grassal, Pier Luigi and Ling, He and Aberman, Koki and Olszewski, Kyle and Prakash, Mim and Tewari, Ayush and Valentin, Julien and Theobalt, Christian and Kemelmacher-Shlizerman, Ira and Thies, Justus},
    title = {Meta Avatars: Evolving our hyper-realistic Codec Avatars with Gaussian splats},
    year = {2024},
    month = {Mar},
    url = {https://tech.meta.com/research/publications/meta-avatars-evolving-our-hyper-realistic-codec-avatars-with-gaussian-splats/},
    note = {Meta Tech Blog}
}

@article{Chen2023HumanGaussian,
  author    = {Chen, Ang and Song, Sida and Wang, Yichun and Xu, Zipei and Liu, Shuai and Chen, ZongSheng and Bao, Hujun and Zhou, Xiaowei and Xu, Dong},
  title     = {HumanGaussian: Text-Driven Controllable Human Image Generation with 3D Gaussians},
  journal   = {arXiv preprint arXiv:2311.17061},
  year      = {2023}
}

@online{Kunder2023HuggingFaceSplatting,
  author    = {Kunder, A. B.},
  title     = {Gaussian Splatting Explained},
  year      = {2023},
  month     = {Aug},
  url       = {https://huggingface.co/blog/gaussian-splatting}
}

@article{Schoenberger2016SFM,
  author    = {Schoenberger, Johannes L. and Frahm, Jan-Michael},
  title     = {Structure-from-Motion Revisited},
  journal   = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@online{Weng2020NeRFJAX,
  author    = {Weng, Lilian},
  title     = {NeRF: Neural Radiance Field in JAX (Part 1: Model and Rendering)},
  year      = {2020},
  month     = {Jul},
  url       = {https://lilianweng.github.io/posts/2020-07-12-nerf/},
  note      = {Lil'Log Blog Post}
}
content_copy
download
Use code with caution.
Bibtex
content_copy
download
Use code with caution.
