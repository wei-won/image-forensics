% Copy-Move Forgery Database with Similar but Genuine Objects %

The Copy-Move Forgery Database with Similar but Genuine Objects (COVERAGE) accompanies the following publication: 
"COVERAGE - A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION," IEEE International Conference on Image Processing (ICIP), 2016

To use:
All data is subject to copyright and may only be used for non-commercial research. 

In case of use, please cite our publication:
B. Wen, Y. Zhu, R. Subramanian, T. Ng, X. Shen, and S. Winkler, "COVERAGE - A Novel Database for Copy-Move Forgery Detection," in Proc. IEEE Int. Conf. Image Processing (ICIP), 2016.

Bibtex:
@inproceedings{wen2016,
  author={Wen, Bihan and Zhu, Ye and Subramanian, Ramanathan and Ng, Tian-Tsong and Shen, Xuanjing and Winkler, Stefan},
  title={COVERAGE - A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION},
  year={2016},
  booktitle={IEEE International Conference on Image processing (ICIP)}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
--------------------------------------------------------------
Data:
Image 
	Original image 			- 	image/*.tif
	Tempered image 			- 	image/*t.tif
Mask
	Duplicated region		- 	mask/*copy.tif
	SGO region (covered) 		- 	mask/*paste.tif
	Forged region 			- 	mask/*forged.tif


* ranges from 1 to 100, denoting 100 pairs of original and forged images
--------------------------------------------------------------
Label / Annotation:
Tampering factor 			- 	label/TFlabel.mat
	- 1 : rotation
	- 2 : scaling
	- 3 : translation
	- 4 : illumination
	- 5 : free-form
	- 6 : combination
	
Tampering Level 			- 	label/Tlevel.mat
	- * only provided for simple tampering (type 1-3)
	- negative number = anti-clockwise rotation
	
Forged Region PSNR 			- 	label/fPSNR.mat
	- * only provided for complex tampering (type 4-6)
	
Forgery Edge Abnormality		- 	label/FEA.mat
	- * only provided for complex tampering (type 4-6)	
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Update Sep 27, 2017
We fixed some Original / Tempered image label errors in the COVERAGE database. 
For users who downloaded the COVERAGE images before, please kindly update them from our oneDrive space.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
COVERAGE description:

COVERAGE contains copymove forged (CMFD) images and their originals with similar but genuine objects (SGOs). COVERAGE is designed to highlight and address tamper detection ambiguity of popular methods, caused by self-similarity within natural images. In COVERAGE, forged¨Coriginal pairs are annotated with (i) the duplicated and forged region masks, and (ii) the tampering factor/similarity metric. For benchmarking, forgery quality is evaluated using (i) computer vision-based methods, and (ii) human detection performance.

Download Dataset at:
https://github.com/wenbihan/coverage

Contact Bihan Wen (bihan.wen.uiuc@gmail.com) for any questions.