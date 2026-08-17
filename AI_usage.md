# AI Usage Documentation — COMP3710 Lab Demo 1

This file documents all AI interactions used while completing Lab Demo 1 (Fractals), per the Fair AI Usage requirements in the Demonstrations Marking Scheme. Each entry records the exact prompt, what the AI produced, whether it worked, and what I evaluated, changed, or fixed myself.

---

## Part 1 — Gaussian, Sine, Gabor Filter

### 1. Generate 2D Gaussian (Numpy)
**Prompt:** "Generate a Python script to plot a 2D Gaussian function using Numpy and Matplotlib"

**Result:** Worked first try. Produced a normalized 2D Gaussian PDF (peak ≈ 0.159) with axis labels and a colorbar, versus my own unnormalized version from the lab sheet (peak = 1.0, using `torch.exp(-(x**2+y**2)/2.0)`). I identified this scaling difference as a deliberate, valid choice by the AI (proper probability density normalization) rather than an error. Also noticed a minor `contourf` edge-rendering artifact near the plot boundary — cosmetic only, not fixed.

### 2. Convert Gaussian script to PyTorch
**Prompt:** "Convert this script to use PyTorch tensors instead of Numpy arrays, and run on GPU if available"

**Result:** Worked first try. Rebuilt the coordinate grid using `torch.linspace` + `torch.meshgrid` directly on `device`, instead of building it in Numpy first and copying it over afterward (the lab sheet's `mgrid` approach) — this is more efficient since it avoids an unnecessary CPU→GPU transfer. Kept the same normalized PDF formula as its previous answer. Compared this against my own `mgrid`-based version from the lab sheet: both are mathematically correct, and differ only in normalization convention and grid-construction method.

### 3. Generate 2D sine function (linear/diagonal)
**Prompt:** "Using PyTorch, create a 2D sine function where the sine's angle depends on both x and y pixel coordinates, and plot it to show stripe patterns"

**Result:** Worked first try. Produced diagonal stripes using `angle = kx*X + ky*Y` (a linear combination of x and y), with `kx = ky = 0.1` controlling stripe frequency and direction. I confirmed this linear form is the correct interpretation to use for the Gabor filter step, since real Gabor filters use oriented/directional sinusoids, not radial ripples.

### 4. Generate 2D sine function (radial/concentric rings)
**Prompt:** "Using PyTorch, create a 2D sine function where the sine's angle depends on both x and y pixel coordinates, and plot it to show concentric rings patterns"

**Result:** The AI gave `sqrt((X - cx)**2 + (Y - cy)**2)` as the angle — not exactly the form I wanted. I worked out myself that using `X**2 + Y**2` directly (without the square root) was the right formula, then diagnosed and fixed a separate aliasing problem on my own: the raw pixel-index grid (0–512, uncentred) caused the local frequency of `sin(x²+y²)` to exceed the Nyquist sampling limit near the image edges, producing a checkerboard/moiré pattern instead of clean rings. Fixed by centring the coordinate grid on zero (`- height//2`) and scaling the angle by a small constant (`k = 0.0005`) to keep the oscillation frequency low enough to be represented correctly across the whole image.

### 5. Gabor filter (Gaussian × sine)
**Not AI-generated.** Built myself by multiplying the Gaussian (`z`) and sine (`z_sine`) tensors elementwise. Verified both tensors were the same shape (800×800) before multiplying, to avoid a silent broadcasting error. Result shows correctly localized, oriented fringes consistent with a Gabor filter / mammalian receptive field model, as described in the lab sheet.

---

## Part 2 — Mandelbrot Set, Zoom, Julia Set

### 6. Generate Mandelbrot set on GPU
**Prompt:** "Write PyTorch code to compute the Mandelbrot set on GPU using 200 iterations and plot it with a colour map."

**Result:** Worked first try, correctly vectorized (no per-pixel Python loop — the whole `z = z*z + c` update runs as a single tensor operation across the grid each iteration). Used escape radius 2 (`torch.abs(z) > 2`), versus the lab sheet's own code which effectively uses radius 4 — both are valid divergence thresholds; radius 2 is the more standard textbook value. Used Matplotlib's built-in `magma` colormap directly on the raw iteration counts, versus the lab sheet's custom cyclic RGB colouring function — a different visual style applied to the same, correct underlying iteration data. Confirmed device selection correctly falls back to CPU on my machine (no CUDA GPU available), consistent with the tutor's guidance that Demo 1 does not require GPU/HPC access.

### High-resolution zoom
**Not AI-generated — self-directed debugging.** My first attempt zoomed into a region (real −0.75 to −0.65, imaginary −0.05 to 0.05) that turned out to be almost entirely inside the Mandelbrot set's main cardioid, producing an almost entirely black image. I diagnosed this myself: the fractal's detailed structure only exists at the boundary between convergence and divergence, not in the solid interior. Re-selected a boundary region (the "seahorse valley," real −0.77 to −0.73, imaginary 0.08 to 0.12) with a much finer grid step (0.00005), which produced the expected dense spiral/filament detail.

### Julia set
**Not AI-generated.** Converted my own Mandelbrot code to a Julia set myself by swapping which quantity is fixed and which varies per pixel: in the Mandelbrot set, `c` varies per pixel and `z₀ = 0` is fixed; in a Julia set, `c` is a single fixed constant and `z₀` varies per pixel (the coordinate grid itself becomes the starting point). Tried several classic constants search from https://vanderbei.princeton.edu/my_julia.html (`-0.8+0.156j`, `-0.4+0.6j`, `0.285+0.01j`, `-0.70176-0.3842j`, `-0.999+0.3j`) and selected `-0.8+0.156j` for the final result, which produced a clean dendrite-like Julia set shape.

---

## Part 3 — Barnsley's Fern via the Chaos Game (Textbook Chapter 6.6)

### Fractal choice
Selected from the lab sheet's pre-approved textbook list (Chapter 6.6, "Program of the Chapter: Chaos Game for the Fern"). Chosen because it uses fundamentally different mathematics from the Mandelbrot/Julia sets covered earlier (random iterated function systems, not complex-plane divergence), satisfying the requirement that Part 3's fractal not be too similar to those already covered.

### Implementation approach
**Not directly AI-generated — designed with AI-assisted discussion of the parallelisation strategy.** 
**Prompt:** "I am planning to generate a fractals and the title is from the textbook Fractals from the Classroom, Chapter 6.6 - Program of the Chapter: Chaos Game for the Fern. I was thinking about parallelism in generating this fractal rather than those traditional method which generating one by one, require heavy computational. Can you give me some idea about this?"

**Result:** Worked. The core idea (run many independent "chaos game" walkers simultaneously as a single batched tensor, instead of one walker run for many sequential steps) was developed through discussion of how to make the classic single-point chaos game algorithm GPU-parallel. Implemented using the standard Barnsley fern IFS coefficients (4 affine maps with fixed probabilities: 0.01 stem, 0.85 main leaflets, 0.07 + 0.07 large left/right leaflets).

Key implementation details I'm responsible for and can explain:
- `torch.bucketize(r, cum_p)` vectorizes the random selection of which of the 4 affine maps each of the 50,000 parallel walkers uses at each step — replacing what would otherwise be a per-walker conditional check.
- The affine transform itself (`x, y = ai*x + bi*y + ei, ci*x + di*y + fi`) is applied to all 50,000 walkers' current positions in a single tensor operation per iteration step.
- A 20-step warmup period is discarded before collecting points, to let walkers settle onto the attractor before recording their trajectory.

### Fractal dimension analysis (substantial analysis requirement)
Implemented box-counting dimension estimation: overlaying grids of decreasing box size over the generated point cloud, counting occupied boxes at each scale, and fitting a line to log(N boxes) vs. log(1/box size) — the slope of this line is the estimated fractal dimension.

I estimated the fern's box-counting dimension at 1.814. Unlike the Sierpinski triangle, which has an exact closed-form dimension because its IFS maps are simple similarities, the fern's maps involve non-uniform scaling and shearing, so there's no equally clean formula to check against — box-counting is genuinely the right empirical approach here, and my log-log plot shows a strongly linear fit, which is the real evidence the estimate is reliable.

### Additional visualisation
Produced a second rendering of the same fern, coloured by iteration step (`cmap='viridis'`) rather than a single flat colour, to show the growth order of the attractor — addresses the rubric's suggestion to incorporate different visualisations/colours as part of the substantial analysis requirement.
