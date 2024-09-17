# Presentation for the MMA team excursion meeting Sept. 2024

The input files are:
- `config.yaml` which sets the slides content
- `default_styling/config.yaml` which sets the default theme values
- `meta/config.yaml` which sets presentation meta data 
- `data` which holds CSV data for plots
- `images` which holds image and video files needed

Requirements:
- [`manim-present`](https://pypi.org/project/manim-present/)
- [`npm` and `NodeJS`](https://github.com/nvm-sh/nvm)

To produce the presentation:
```bash
./produce.sh
```

Which will generate (these are the files that need to be published):
- `YamlPresentation_assets` for the presentation assets
- `index.hml` as a main entry point
It will also log the run in `outputs` folder
