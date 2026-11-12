"""Twelve 1D kernels and the original sixteen thread-count actions."""
import math

THREAD_COUNTS = tuple(range(64, 1025, 64))
KINDS = ('vector_add', 'segmented_sum', 'periodic_tile', 'repeat_vector',
         'matrix_transpose', 'mean', 'variance', 'nbody',
         'binarize_weights', 'binarize_input', 'softmax', 'accumulate')
ELEMENTWISE = ('vector_add', 'periodic_tile', 'repeat_vector', 'matrix_transpose')
TENSOR_OUTPUT = ('binarize_weights', 'binarize_input', 'softmax')
GROUPS = dict(vector_add='lscat_repo_10047', segmented_sum='lscat_repo_10040',
              periodic_tile='lscat_repo_10073', repeat_vector='lscat_repo_1018',
              matrix_transpose='lscat_repo_10194', mean='lscat_repo_10194',
              variance='lscat_repo_10194', nbody='nvidia_nbody_adapted',
              **{name:'lscat_repo_10194' for name in (*TENSOR_OUTPUT,'accumulate')})
SMOKE = {name: (17, 1, 1) for name in ELEMENTWISE}
SMOKE.update({name: (35, 1, 67) for name in ('segmented_sum', 'mean', 'variance')})
SMOKE['nbody'] = (259, 1, 1)
SMOKE.update({name:(35,1,67) for name in (*TENSOR_OUTPUT,'accumulate')})
REDUCTION = [(256,1,1024), (1024,1,4096), (2048,1,16384), (4096,1,65536),
             (1024,1,262144), (256,1,1048576), (128,1,2097152),
             (64,1,4194304), (32,1,8388608), (16,1,16777216)]
LADDERS = {name: [(n,1,1) for n in (240,1024,2048,4096,8192,12288,16384,22528,32768)]
           for name in ELEMENTWISE}
LADDERS.update({name: REDUCTION for name in ('segmented_sum', 'mean', 'variance')})
LADDERS['accumulate'] = REDUCTION
# These kernels write M*K outputs as well as reading M*K inputs.
LADDERS.update({name:[(128,1,1024),(256,1,4096),(512,1,16384),(1024,1,65536),
                      (512,1,262144),(128,1,1048576),(64,1,2097152),
                      (32,1,4194304),(16,1,8388608),(8,1,16777216)]
                for name in TENSOR_OUTPUT})
LADDERS['nbody'] = [(n,1,1) for n in (8192,16384,32768,65536,131072,262144,
                                   524288,786432,1048576,1572864,2097152,4194304)]

def counts(name, m, n, k):
    if name not in KINDS or any(type(x) is not int or x < 1 for x in (m,n,k)):
        raise ValueError('Invalid workload')
    if name == 'nbody':
        if not 2 <= m <= 4194304 or (n,k) != (1,1):
            raise ValueError('Invalid body count')
        return 4*m, 0, 3*m
    if name in ELEMENTWISE:
        if m > 32768 or (n,k) != (1,1):
            raise ValueError('Invalid square output dimension')
        out = m*m
        return (221 if name == 'periodic_tile' else m if name == 'repeat_vector' else out,
                out if name == 'vector_add' else 0, out)
    if m > 16384 or n != 1 or not 2 <= k <= 16777216 or m*k > 2147483647:
        raise ValueError('Dimensions exceed reviewed reduction/indexing bounds')
    return m*k, m if name == 'variance' else 0, m*k if name in TENSOR_OUTPUT else m

def device_bytes(name, m, n, k):
    return 4*(sum(counts(name,m,n,k))+32)

def actions(name, m, n, k, device):
    _,_,out = counts(name,m,n,k)
    limit = min(device['max_threads_per_block'], device['kernels'][name]['max_threads_per_block'])
    if limit < 1024 or device['max_block_dim_x'] < 1024:
        raise ValueError('All 16 configurations are required; device/kernel cannot launch 1024 threads: '+name)
    work = out if name in ELEMENTWISE else m
    result = [dict(block_x=b, block_y=1, grid_x=math.ceil(work/b), grid_y=1,
                   grid_policy='coverage') for b in THREAD_COUNTS]
    if any(a['grid_x'] > device['max_grid_dim_x'] for a in result):
        raise ValueError('Coverage grid exceeds device limit: '+name)
    return result

def reference(options):
    return next(a for a in options if a['block_x'] == 256 and a['block_y'] == 1)

def tier(ms):
    if ms < 1: return 'sub_ms_control'
    if ms < 5: return '1_to_5_ms'
    if ms < 50: return '5_to_50_ms'
    if ms < 500: return '50_to_500_ms'
    if ms < 1000: return '0.5_to_1_second'
    if ms <= 2000: return '1_to_2_seconds'
    return 'over_2_seconds'
