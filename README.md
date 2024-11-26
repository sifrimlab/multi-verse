
<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT Liscence][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
 <!-- <img src="" alt="logo" align="center"> -->
  <h3 align="center"> Multi-verse</h3>

  <p align="center">
    A package for comparing MOFA, MOWGLI, MultiVI, and PCA on multimodal datasets, providing scIB metrics and UMAP visualizations.

    <br />
    <a href="https://github.com/sifrimlab/multi-verse/issues">Report Bug</a>
    ·
    <a href="https://github.com/sifrimlab/multi-verse/pulls">Add Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <li><a href="#practicalities">Practicalities</a></li>
    <li><a href="#contributing">Contributing</a></li>
   <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Multi-verse is a Python package designed to facilitate the comparison of multimodal data integration methods, specifically MOFA, MOWGLI, MultiVI, and PCA. By leveraging scIB metrics and generating UMAP visualizations, this package enables researchers to assess and visualize the performance of these methods on their datasets.

Key features:
- Supports comparison of four major methods: [MOFA](https://biofam.github.io/MOFA2/), [MOWGLI](https://mowgli.readthedocs.io/en/latest/index.html), [MultiVI](https://docs.scvi-tools.org/en/1.2.0/user_guide/models/multivi.html), and PCA.
- Provides scIB metrics for integration performance evaluation.
- Generates UMAP visualizations for easy interpretation of results.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow steps below.

### Prerequisites

It is recommended to create a new virtual enviroment with [conda](https://www.anaconda.com/).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sifrimlab/multi-verse.git
   cd multi-verse
   ```

2. Create a new conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate multi_verse
    ```

## Usage
1. To run the script, provide a configuration JSON file as an argument. The configuration file should include all necessary settings for the methods and metrics you want to compare. See config.json for example structure.

2. Run the code (with exmaple config.json file):
    ```bash
    python main.py config.json
    ```


<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!--LICENSE -->
## License

Distributed under the GPL-3 License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact
Project Link: https://github.com/sifrimlab/multi-verse


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/sifrimlab/multi-verse.svg?style=for-the-badge
[contributors-url]: https://github.com/sifrimlab/multi-verse/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sifrimlab/multi-verse.svg?style=for-the-badge
[forks-url]: https://github.com/sifrimlab/multi-verse/network/members
[stars-shield]: https://img.shields.io/github/stars/sifrimlab/multi-verse.svg?style=for-the-badge
[stars-url]: https://github.com/sifrimlab/multi-verse/stargazers
[issues-shield]: https://img.shields.io/github/issues/sifrimlab/multi-verse.svg?style=for-the-badge
[issues-url]: https://github.com/sifrimlab/multi-verse/issues
[license-shield]: https://img.shields.io/badge/license-GPL--3.0--only-green?style=for-the-badge
[license-url]: https://github.com/sifrimlab/multi-verse/LICENSE
