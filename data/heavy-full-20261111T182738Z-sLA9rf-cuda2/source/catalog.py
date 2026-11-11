"""Heavy workload contracts and legal action spaces; Python standard library only."""
import math

KINDS = ('gemm', 'segmented_sum', 'mean', 'variance', 'syrk', 'nbody')
GROUPS = dict(gemm='lscat_repo_104', segmented_sum='lscat_repo_10040',
              mean='lscat_repo_10194', variance='lscat_repo_10194',
              syrk='polybench_syrk', nbody='nvidia_nbody_adapted')
SMOKE = dict(gemm=(17,19,23), segmented_sum=(35,1,67), mean=(35,1,67),
             variance=(35,1,67), syrk=(17,17,23), nbody=(259,1,1))
REDUCTION = [(256,1,1024), (1024,1,4096), (2048,1,16384), (4096,1,65536),
             (1024,1,262144), (256,1,1048576)]
LADDERS = {
    'gemm': [(512,512,256),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),
             (4096,4096,8192),(8192,8192,8192),(8192,8192,16384)],
    'syrk': [(512,512,512),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),
             (4096,4096,8192),(8192,8192,8192),(8192,8192,16384)],
    'nbody': [(n,1,1) for n in (8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304)],
    **{name: REDUCTION for name in ('segmented_sum','mean','variance')},
}

def counts(name, m, n, k):
    if name not in KINDS or any(type(x) is not int or x<1 for x in (m,n,k)):
        raise ValueError('Invalid workload')
    if name=='nbody':
        if not 2<=m<=4194304 or (n,k)!=(1,1): raise ValueError('Invalid body count')
        return 4*m,0,3*m
    if m>16384 or n>16384 or k>1048576 or m*k>2147483647 or m*n>2147483647:
        raise ValueError('Dimensions exceed reviewed indexing bounds')
    if name in ('gemm','syrk'):
        if k*n>2147483647 or (name=='syrk' and m!=n): raise ValueError('Invalid matrix shape')
        return m*k, k*n if name=='gemm' else 0, m*n
    if n!=1 or k<2: raise ValueError('Invalid reduction shape')
    return m*k,m if name=='variance' else 0,m

def device_bytes(name,m,n,k):
    return 4*(sum(counts(name,m,n,k))+32)

def actions(name,m,n,k,device,smoke=False):
    counts(name,m,n,k)
    blocks = [(8,8),(16,8),(8,16),(16,16),(32,8),(8,32),(32,16),(16,32),(32,32)] if name in ('gemm','syrk') else [(b,1) for b in range(64,1025,64)]
    limit=min(device['max_threads_per_block'],device['kernels'][name]['max_threads_per_block'])
    result=[];seen=set()
    for bx,by in blocks:
        if bx*by>limit or bx>device['max_block_dim_x'] or by>device['max_block_dim_y']:continue
        if name in ('gemm','syrk'):
            gx,gy=math.ceil(n/bx),math.ceil(m/by)
            grids=[('coverage',gx,gy),('extra_x',2*gx,gy),('extra_y',gx,2*gy)]
        else:
            gx=math.ceil(m/bx);grids=[('coverage',gx,1)]
            if name=='nbody':
                grids += [('sm_'+str(c),min(gx,device['sm_count']*c),1) for c in (1,2,4,8)]
                if smoke:grids.append(('smoke_cap1',1,1))
            else:
                grids.append(('extra_x',2*gx,1))
                if name in ('mean','variance'):
                    grids += [('reshape_y'+str(y),math.ceil(gx/y),y) for y in (2,8) if gx>=y]
        for policy,gx,gy in grids:
            key=bx,by,gx,gy
            if key in seen or gx>device['max_grid_dim_x'] or gy>device['max_grid_dim_y']:continue
            seen.add(key);result.append(dict(block_x=bx,block_y=by,grid_x=gx,grid_y=gy,grid_policy=policy))
    if not result:raise ValueError('No legal actions for '+name)
    return result

def reference(options):
    want=(16,16) if any(r['block_y']>1 for r in options) else (256,1)
    candidates=[r for r in options if r['grid_policy']=='coverage']
    return next((r for r in candidates if (r['block_x'],r['block_y'])==want),None) or min(candidates,key=lambda r:abs(r['block_x']*r['block_y']-256))

def tier(ms):
    if ms<1:return 'sub_ms_control'
    if ms<5:return '1_to_5_ms'
    if ms<50:return '5_to_50_ms'
    if ms<500:return '50_to_500_ms'
    if ms<1000:return '0.5_to_1_second'
    if ms<=5000:return '1_to_5_seconds'
    return 'over_5_seconds'
