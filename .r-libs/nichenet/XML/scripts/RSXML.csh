if(`test -n "-L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -lxml2 -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -lz -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -liconv -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -licuuc"`) then

if(${?LD_LIBRARY_PATH}) then
    setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:-L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -lxml2 -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -lz -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -liconv -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -licuuc
else
   setenv LD_LIBRARY_PATH -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -lxml2 -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -lz -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -liconv -L/Users/k.h.prange/miniconda3/envs/scOmnom_env/lib -licuuc
endif

endif
