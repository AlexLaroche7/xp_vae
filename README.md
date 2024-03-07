# Closing the stellar labels gap: Stellar label independent evidence for [$`\alpha`$/M] information in Gaia BP/RP spectra
Code for Laroche &amp; Speagle 2024 (in prep)

## Abstract

Data-driven models for stellar spectra which depend on stellar labels suffer from label systematics which decrease model performance: the “stellar labels gap”. To close the stellar labels gap, we present a stellar label independent model for *Gaia* BP/RP (XP) spectra. We develop a novel implementation of a variational auto-encoder; a \emph{scatter} VAE, which learns to generate an XP spectrum and accompanying intrinsic scatter without any reliance on stellar labels. We demonstrate that our model achieves better XP spectra reconstructions than stellar label dependent models. We find that our model learns stellar properties directly from the data itself. We then apply our model to giant stars from an APOGEE/XP cross-match to study the [$`\alpha`$/M] information in *Gaia* XP. We provide unambiguous evidence that the XP spectra contain meaningful [$`\alpha`$/M] information by demonstrating that our model learns the $`\alpha`$-bimodality *without relying on stellar label correlations*. We publicly release our trained model and codebase. Importantly, our stellar label independent model can be implemented for any/all XP spectra because our model performance scales with training object density, not training label density.

![model arch](https://github.com/AlexLaroche7/xp_vae/blob/main/figures/svae_arch.png)

## Reproducing figures
If you would like to access the data used in this work to reproduce our results, feel free to contact me via [email](mailto:alex.laroche@mail.utoronto.ca).
