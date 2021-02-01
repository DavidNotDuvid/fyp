# Final year project for NUS computational biology project 2019/2020.
# Quantitative biology with deep learning image segmentation

# Introduction
Muscle tissues are critical to the everyday functioning of living organisms. For humans, they allow us to perform day-to-day activities, from drinking coffee to writing and even reading this paper. Muscle tissues are made out of muscle fibres which form during a process known as myogenesis, which occurs in the embryo shortly after gastrulation (Kimmel, C., 1995).
Muscle tissues are formed in response to signals from fibroblast growth factor, serum and calcium. In the presence of fibroblast growth factor, myoblasts fuse and form into
multinucleated myotubes, which form the basis of muscle tissue. However, the exactmechanism of this event is still unknown (Kim, J. et al 2015)

# The problem
To understand the process of myogenesis, we must be able to follow the myoblasts throughout their development: this spans from subcellular spatial scales (e.g. during fusion)
to 100 microns (whole tissue) and temporal scales from seconds to hours. However, following this process is difficult due to the immense amount of data and activity occurring at
any given time and the 3-dimensional spatial scale where this is occurring on. We try to achieve this via accurate nuclei segmentation, an essential process in the quantitative study of the dynamics in a biological system (Meijering, E. 2012). However, developing zebrafishm somites are exceptionally difficult to segment due to its dense cell environment , making it hard to clearly differentiate closely compacted nuclei along with the sheer amountof background noise present in the biological system. The inability to differentiate nuclei hinders the ability to fully follow fusion events. Therefore, new techniques are required to analyse this complex four-dimensional problem. Here, we utilise deep learning approaches.

# The solution

U-net (Ronneberger, O., et al 2015) is a Convolutional neural network (CNN) architecture that is designed for biomedical data. A pipeline utilising this architecture was designed and implemented to segment our data. A unique feature of the U-NET architecture was that it allowed for training with only a small dataset. This is advantageous especially in the field of biology as minimum man hours are needed to curate ground truth data. This allows for maximum efficiency and cost to return. The pipeline was implemented using Tensorflow using Jupyter Notebook. A U-NET model was trained from scratch and tested on our data. As seen in Figure 6. The homegrown U-net model performed much better compared to the existing model discussed previously. Unlike the aforementioned methods, the model could differentiate compact nuclei. In addition, it was able to partially separate somatic nucleus from notochord and skin nucleus while performing complete and whole segmentation of the nucleus. This is ideal as it reduces the work needed to manually distinguish nucleus of interest from the rest of the nucleus.

