This dir, `model/`, will contain our encoder / decoder / model. Then we can import them

## Our Modifications to VFIT

1. A loss function that factors in 2 outputs: the inputs, and the temporally flipped inputs

2. Our model generalizes to more than 2 input frames.

3. Concatenated time encodings to the latent representation of the input frames before passing to them to the SynBlocks.
