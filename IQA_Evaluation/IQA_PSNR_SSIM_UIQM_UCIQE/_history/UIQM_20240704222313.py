import tensorflow as tf

  # UIQM
def eme_tf(ch,blocksize=8):

        num_x = tf.math.ceil(ch.shape[0] / blocksize)
        num_y = tf.math.ceil(ch.shape[1] / blocksize)
        
        eme = tf.constant(0.0, dtype=tf.float32)
        w = 2.0 / (num_x * num_y)
        for i in range(blocksize):
          for j in range(blocksize):
            block = ch[:, i::blocksize, j::blocksize, :]
            blockmin = tf.reduce_min(block, axis=[0, 1, 2], keepdims=True)
            blockmax = tf.reduce_max(block, axis=[0, 1, 2], keepdims=True)
            blockmin = tf.maximum(blockmin, 1e-12)  # avoid division by zero
            blockmax = tf.maximum(blockmax, 1e-12)  # avoid division by zero
            eme += w * tf.math.log(blockmax / blockmin)               
        return eme

def plipsum(i,j,gamma=1026.0):
        return i + j - i * j / gamma

def plipsub(i,j,k=1026.0):
        return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026.0):
      return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

        num_x = tf.math.ceil(ch.shape[0] / blocksize)
        num_y = tf.math.ceil(ch.shape[1] / blocksize)
        
        s = tf.constant(0.0, dtype=tf.float32)
        w = 1.0 / (num_x * num_y)

        for i in range(blocksize):
          for j in range(blocksize):
            block = ch[:, i::blocksize, j::blocksize, :]
            blockmin = tf.reduce_min(block, axis=[0, 1, 2], keepdims=True)
            blockmax = tf.reduce_max(block, axis=[0, 1, 2], keepdims=True)

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)
            m = tf.where(tf.equal(bottom, 0), tf.zeros_like(bottom), top / bottom)
            logameee += w * m * tf.math.log(tf.maximum(m, 1e-12))
        return plipmult(tf.reduce_mean(logameee), 1.0)

def getUIQM(image):
        gray = tf.image.rgb_to_grayscale(image)
      
        # UIQM
        p1 = 0.0282
        p2 = 0.2953
        p3 = 3.5753

        #1st term UICM
        rg = image[:,:,0] - image[:,:,1]
        yb = (image[:,:,0] + image[:,:,1]) / 2 - image[:,:,2]
        rgl = tf.sort(rg,axis=None)
        ybl = tf.sort(yb,axis=None)
        al1 = 0.1
        al2 = 0.1
        T1 = tf.cast(al1 * tf.size(rgl),tf.int32)
        T2 = tf.cast(al2 * tf.size(rgl),tf.int32)
        rgl_tr = rgl[T1:-T2]
        ybl_tr = ybl[T1:-T2]

        urg = tf.reduce_mean(rgl_tr)
        s2rg = tf.reduce_mean(tf.square(rgl_tr - urg))
        uyb = tf.reduce_mean(ybl_tr)
        s2yb = tf.reduce_mean(tf.square(ybl_tr- uyb))

        uicm =-0.0268 * tf.sqrt(urg**2 + uyb**2) + 0.1586 * tf.sqrt(s2rg + s2yb)

        #2nd term UISM (k1k2=8x8)
        Rsobel = image[:,:,0] * tf.image.sobel_edges(image[:,:,0:1])
        Gsobel = image[:,:,1] * tf.image.sobel_edges(image[:,:,1:2])
        Bsobel = image[:,:,2] * tf.image.sobel_edges(image[:,:,2:3])

        Rsobel=tf.round(Rsobel)
        Gsobel=tf.round(Gsobel)
        Bsobel=tf.round(Bsobel)

        Reme = eme_tf(Rsobel)
        Geme = eme_tf(Gsobel)
        Beme = eme_tf(Bsobel)

        uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

        #3rd term UIConM
        uiconm = logamee(gray)

        uiqm = p1 * uicm + p2 * uism + p3 * uiconm
        print("UIQM:", uiqm)
        
        return uiqm
