# DANCE

Code of Domain Adaptation of Networks for Camera Pose Estimation: Learning Camera Pose Estimation Without PoseLabels by Jack Langerman, Ziming Qiu, Gabor Soros, David Sebok, Yao Wang, Howard Huang
Nokia Bell Labs and New York University, 2020

Paper link: [arxiv](https://arxiv.org/pdf/2111.14741.pdf)

Dataset link: [Dataport](https://ieee-dataport.org/documents/bell-labs-robot-garage-dance-domain-adaptation-networks-camera-pose-estimation).
In the dataset, we have training images with 100,000 labeled rendered images and 28411 unlabeled real camera images. We also have the validation set (1637 labeled real camera images) and test set (2104 labeled real camera images).

Bibtex:
```
@article{DBLP:journals/corr/abs-2111-14741,
  author    = {Jack Langerman and Ziming Qiu and G{\'{a}}bor S{\"{o}}r{\"{o}}s and D{\'{a}}vid Sebok and Yao Wang and Howard Huang},
  title     = {Domain Adaptation of Networks for Camera Pose Estimation: Learning
               Camera Pose Estimation Without Pose Labels},
  journal   = {CoRR},
  volume    = {abs/2111.14741},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.14741},
  eprinttype = {arXiv},
  eprint    = {2111.14741},
  timestamp = {Wed, 01 Dec 2021 15:16:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-14741.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# Training

Training:

(1) run `histogram_match.ipynb` to preprocess the training rendered images.

(2) going into cut folder, use the `prepare_dataset.ipynb` to prepare training data, then run `run_py_job.sbatch` to train the CUT GAN model.

(3) run `train_init_scr_cut.ipynb` to train the final scene coordinate regression model.

# Testing

(1) run `test.ipynb`


