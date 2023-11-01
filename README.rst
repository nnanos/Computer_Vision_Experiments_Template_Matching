=======================================================================
Computer Vision Experiments Template matching
=======================================================================



The problem of Template matching goes back to the basic problem of computer vision,
the correspondence problem (i.e. given two images find the geometric
transformation that connects them). This problem is solved either with feature
based techniques or with correlation based techniques. In this repo, the problem
will be approached with correlation techniques. More specifically we will use
the operation of cross-correlation which is implemented as convolution of one
image with another flipped which as we know is implemented quickly with
FFT algorithm. 

**In this repo I also use the pyramids_implementation script that is implemented by me in this repo** 

**Algorithm implemented:**

#. Creating a Laplacian pyramid from the gaussian of the scene image 
   (image containing the template).

#. For each level of the previous pyramid take its convolution with the flipped
   image template (in essence the response will be the cross-correlation of the
   two images) and then from each calculation finding the highest response
   (max element of each response).

#. From the list of maximum responses we find the maximum where its index
   of the maximum in the list will show us the correct scale where the teplate
   is and so we find its place on that scale.

#. After we have found the position on the correct scale we draw a square with
   center this position (and these dimensions of the template) and then we
   reconstruct the image at the original scale (with the square drawn) .






Experiments
============

* Template without Additive Gaussian noise 



.. Image:: /Documentation_Images/cross_cor_res1.png


* Template matching result


.. Image:: /Documentation_Images/res1.png



* Template with Additive Gaussian noise


.. Image:: /Documentation_Images/cross_cor_res2.png


* Template matching result
  

.. Image:: /Documentation_Images/res2.png




We notice that even with the noisy pattern the technique with cross correlations
works (for the specific noisy template). The only difference is that the response
to the right level drops from 0.6 to 0.4. Therefore we understand that if we
increase the power of the additive noise to template the more likely the
algorithm will fail to detect the correct scale at which the template is located.




Reproduce the Experiments
============



Free software: MIT license
============
