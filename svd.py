import numpy as np
import imageio;
from skimage.color import rgb2gray; 
import matplotlib.pyplot as plt

img_url = "pb.jpg"
import os
b = os.path.getsize(img_url)


A=imageio.imread(img_url);
#A=rgb2gray(A)
A=np.double(A); 
A=A-np.mean(A);



def compres(A):
	U, S, V = np.linalg.svd(A)
	svdd = np.flip(np.sort(S))

	percent_var = 0.

	sv_kept = 1
	while (percent_var < .95):
		percent_var = np.sum(svdd[:sv_kept]**2)/(np.sum(svdd**2))
		print(percent_var)
		sv_kept += 1

	print("We should keep {} singular values".format(sv_kept))


	U_recon = U[:,0:sv_kept]
	V_recon = V[0:sv_kept,:]
	S_recon = S[:sv_kept]

	A_recon = U_recon@np.diag(S_recon)@V_recon

	return A_recon


R = compres(A.reshape(A.shape[0],A.shape[1]*3))



img_recon = R.reshape(600,900,3)

c = imageio.imwrite(img_url + 'compressed.jpg', img_recon)
d = os.path.getsize('astronaut-gray.jpg')

reduction = (b-d)/d

print("Your file has been reduced by {} percent".format(reduction*100))
fig,ax = plt.subplots(1,2,figsize=(15,7))
ax[1].imshow(A)
ax[1].set_title('Before Compression')
ax[0].imshow(img_recon);
ax[0].set_title('After Compression')

plt.show()


