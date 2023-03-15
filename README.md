## MNIST: one image is normal, others anomalous

Teacher AE as here:
[https://openaccess.thecvf.com/content/CVPR2022W/WiCV/papers/Schneider_Autoencoders_-_A_Comparative_Analysis_in_the_Realm_of_Anomaly_CVPRW_2022_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022W/WiCV/papers/Schneider_Autoencoders_-_A_Comparative_Analysis_in_the_Realm_of_Anomaly_CVPRW_2022_paper.pdf)

Teacher free parameters: 155,981   
Teacher inference time/img on CPU: 32 ms  

Student free parameters:  216   
Student inference time/img on CPU: 1.5 ms  

Compression factor in free parameters: 134.34   
Compression factor in FLOPS: 281.05  

In folders, FASHION_MNIST and MNIST, 
you will find the teachers trained for each label and corresponding students, 
distinguished by their size measured in number of free parameters.

In folders with 'student_normals', you will find incomplete analysis only using a fraction of normal digits for training student.
