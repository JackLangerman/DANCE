# DANCE
The paper link: https://arxiv.org/pdf/2111.14741.pdf

The data set link: https://ieee-dataport.org/documents/bell-labs-robot-garage-dance-domain-adaptation-networks-camera-pose-estimation.
Inside the link, we have train folders with 100,000 labeled rendered images and	28411 unlabeled	real camera images.
We also	have the validation set	(1637 labeled real camera images) and test set (2104 labeled real camera images).

Training:
(1) histogram_match.ipynb to preprocess the training rendered images.

(2) going into cut folder, use the prepare_dataset.ipynb to prepare training data, run run_py_job.sbatch to train the cut based gan model.

(3) train_init_scr_cut.ipynb to train the final SCR model.

Testing:
(1) test.ipynb
