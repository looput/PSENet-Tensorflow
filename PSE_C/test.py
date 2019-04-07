import mylib
import numpy as np

CC = np.array([
    [0, 0, 0],
    [0, 1, 0]
], dtype=np.int32)

Si = np.array([
    [0, 1, 1],
    [0, 1, 0]
], dtype=np.int32)

ps = mylib.PyExpand()
ps.expansion(CC, Si)

# embed_vc = np.array([[[2, 3, 3, 4], [2, 4, 12, 2]],
#                      [[3, 3, 4, 2], [4, 6, 6, 1]]],dtype=np.float)
                    
embed_vc=np.random.rand(640,640*4,16).astype(np.float)
shape=np.array(embed_vc.shape,dtype=np.int32)
seed=np.array([0,0],np.int32)
delta=10.

mask=np.zeros(embed_vc.shape[0:2],np.int32)
pe = mylib.PyRegion()

for i in range(5):
    mask=mask*0
    pe.region_grow(embed_vc,shape,seed,delta,mask)

print(mask)
