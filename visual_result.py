import json
import os

import matplotlib.pyplot as plt

infer_dir='Logs/test/ic15_15'
result_list=os.listdir(infer_dir)

class metric():
    h,p,r=None,None,None
    train_step=None
    train_id=None

evals={}
for index,res_pt in enumerate(result_list):
    result=os.path.join(infer_dir,res_pt,'result.json')
    try:
        with open(result,'r') as f:
            res=json.load(f)
            ev={
                'h':res['method']['hmean'],
                'p':res['method']['precision'],
                'r':res['method']['recall'],

                'train_id':int(res_pt.split('-')[-1].split('_')[1]),
                'train_step':int(res_pt.split('-')[-1].split('_')[0])
            }

            evals['{}'.format(index)]=ev
            # print(index,': ',ev)
    except:
        print('error! don\'t find result')
evals_sort=sorted(evals.values(),key=lambda v: v['train_step']+v['h'])
for value in evals_sort:
    print(value)
# print('max h mean value',max(h))

with plt.style.context('Solarize_Light2'):
    fig,ax=plt.subplots()
    for key, value in evals.items():
        if value['train_id']>0:
            ax.scatter(value['train_step'],value['h'],marker='+',)
            # ax.annotate(str(value['h']*100)[0:4],xy=(value['train_step'],value['h']))
        else:
            ax.scatter(value['train_step'],value['h'],marker='o',)
            ax.annotate(str(value['h']*100)[0:4],xy=(value['train_step'],value['h']))

plt.show()

