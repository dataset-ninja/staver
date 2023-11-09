**StaVer: Stamp Verification** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is applicable or relevant across various domains. 

The dataset consists of 427 images with 377 labeled objects belonging to 1 single class (*stamp*).

Images in the StaVer dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 87 (20% of the total) unlabeled images (i.e. without annotations). There are no pre-defined <i>train/val/test</i> splits in the dataset. Also, dataset includes meta-information provided within tags: ***signature***, ***overlap_with_printed_text*** and ***number_of_stamps***. The dataset was released in 2011 by the Aarhus University (AU), Denmark and German Research Center for Artiﬁcial Intelligence (DFKI), Germany.

<img src="https://github.com/dataset-ninja/staver/raw/main/visualizations/poster.png">
