# Beta-Zero

### Using machine-learning to set climbs on spray walls and system boards.

BetaZero is live and free to use at https://betazero.live!

This is an open-source application which leverages a projected diffusion model to set custom climbs on homewalls and system boards.
#### How it works
* Upload your homewall and add holds
* Generate Climbs using the diffusion model, or set your own climbs manually
* Save climbs to the database, and view previously set climbs

### Additional features
* User authentication with Clerk
* Homewall privacy settings (Public, Private, Unlisted)
* Support for public system boards (Aurora, Kilter, Moon)

If you have any questions, feedback, or feature requests, let me know here! You can also fill out this feedback form: https://docs.google.com/forms/d/e/1FAIpQLSeYDIel5MMjj0X3zlXFe4N4FZdUcXadAL5bR-Wjb4W7SVZ5SQ/viewform?usp=header

### BoardLib Dataset

This iteration of BetaZero was trained on the Aurora/Kilter climbs dataset, using the amazing open-source BoardLib API: https://github.com/lemeryfertitta/BoardLib.

### Model-Training (DDPM)

See the write-up about DDPM training here: https://evmojo37.substack.com/p/betazero-v2-a-diffusion-model-for
![GeneratingHoldsWithRoles](https://github.com/user-attachments/assets/c3375bcd-4813-4186-b58e-9d3e23d78c67)

The climbs you see being generated here are completely novel! The coolest part about this model is that it should in theory be able to generate reasonable climbs for *any* system board, provided that the user accurately uploads the board's hold positions.
