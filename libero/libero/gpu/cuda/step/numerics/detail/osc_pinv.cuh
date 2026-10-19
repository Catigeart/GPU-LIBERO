#pragma once

// Each lane owns a row of both column-major operands. Column-pair rotations
// remain ordered, while their row arithmetic and the final inverse cooperate.
// The three independent OSC problems use one full warp and two half warps.
struct OscPseudoinverseWork {
    double columns[36];
    double vectors[36];
};

template<int Width>
static __device__ double pinv_row_sum(double value,int rows,unsigned mask) {
    double result=0.0;
    // Retain the row order of the scalar one-sided Jacobi implementation.
    #pragma unroll 1
    for(int row=0;row<rows;++row)result+=__shfl_sync(mask,value,row,Width);
    return result;
}

template<int N,int Width>
static __device__ __noinline__ void pinv_group(const double* input,double* output,
    OscPseudoinverseWork& work,int* rank,int* ok,unsigned mask) {
    const int lane=threadIdx.x&(Width-1);
    double scale=0.0;
    if(lane<N)for(int col=0;col<N;++col)scale=fmax(scale,fabs(input[lane*6+col]));
    #pragma unroll 1
    for(int row=0;row<N;++row)scale=fmax(scale,__shfl_sync(mask,scale,row,Width));
    for(int z=lane;z<36;z+=Width)output[z]=0.0;
    if(lane==0){*rank=0;*ok=1;}
    if(!(scale>0.0))return;
    if(lane<N)for(int col=0;col<N;++col) {
        work.columns[col*6+lane]=input[lane*6+col]/scale;
        work.vectors[col*6+lane]=col==lane?1.0:0.0;
    }
    bool converged=false;
    #pragma unroll 1
    for(int sweep=0;sweep<64;++sweep) {
        int rotations=0;
        #pragma unroll 1
        for(int p=0;p<N;++p) {
            #pragma unroll 1
            for(int q=p+1;q<N;++q) {
                const double bp=lane<N?work.columns[p*6+lane]:0.0;
                const double bq=lane<N?work.columns[q*6+lane]:0.0;
                const double alpha=pinv_row_sum<Width>(bp*bp,N,mask);
                const double beta=pinv_row_sum<Width>(bq*bq,N,mask);
                const double gamma=pinv_row_sum<Width>(bp*bq,N,mask);
                if(fabs(gamma)>2.220446049250313e-16*sqrt(alpha)*sqrt(beta)) {
                    const double zeta=(beta-alpha)/(2*gamma);
                    const double t=(zeta<0?-1.0:1.0)/(fabs(zeta)+sqrt(1+zeta*zeta));
                    const double c=1/sqrt(1+t*t),s=c*t;
                    if(lane<N) {
                        const double vp=work.vectors[p*6+lane],vq=work.vectors[q*6+lane];
                        work.columns[p*6+lane]=c*bp-s*bq;
                        work.columns[q*6+lane]=s*bp+c*bq;
                        work.vectors[p*6+lane]=c*vp-s*vq;
                        work.vectors[q*6+lane]=s*vp+c*vq;
                    }
                    ++rotations;
                }
            }
        }
        if(rotations==0){converged=true;break;}
    }
    double norms[N]={},singular[N]={},weights[N]={},maxnorm=0.0;
    bool retained[N]={};
    #pragma unroll 1
    for(int p=0;p<N;++p) {
        const double value=lane<N?work.columns[p*6+lane]:0.0;
        norms[p]=pinv_row_sum<Width>(value*value,N,mask);
        singular[p]=sqrt(norms[p]);maxnorm=fmax(maxnorm,singular[p]);
    }
    int retained_count=0;
    for(int p=0;p<N;++p) {
        retained[p]=singular[p]>1e-15*maxnorm;
        if(retained[p]){++retained_count;weights[p]=(1/norms[p])/scale;}
    }
    double maxcorr=0.0;
    #pragma unroll 1
    for(int p=0;p<N;++p) {
        #pragma unroll 1
        for(int q=p+1;q<N;++q)if(retained[p]&&retained[q]) {
            const double bp=lane<N?work.columns[p*6+lane]:0.0;
            const double bq=lane<N?work.columns[q*6+lane]:0.0;
            const double gamma=pinv_row_sum<Width>(bp*bq,N,mask);
            maxcorr=fmax(maxcorr,fabs(gamma)/(singular[p]*singular[q]));
        }
    }
    if(maxcorr<2e-15)converged=true;
    // Reconstructing a cell reads other lanes' rows, unlike the row-owned
    // rotations above. Close those writes before changing ownership.
    __syncwarp(mask);
    for(int z=lane;z<36;z+=Width) {
        const int row=z/6,col=z%6;
        double value=0.0;
        if(row<N&&col<N)for(int k=0;k<N;++k)if(retained[k])
            value+=work.vectors[k*6+row]*work.columns[k*6+col]*weights[k];
        output[z]=value;
    }
    if(lane==0){*rank=retained_count;*ok=converged;}
}

// All world-block threads enter. The caller owns the block publication barrier.
static __device__ void pinv_three_block(const double* full,const double* position,
    const double* orientation,double* full_inverse,double* position_inverse,
    double* orientation_inverse,int* ranks,int* ok) {
    __shared__ OscPseudoinverseWork work[3];
    if(threadIdx.x<32)
        pinv_group<6,32>(full,full_inverse,work[0],ranks,ok,0xffffffffu);
    else if(threadIdx.x<64) {
        const int problem=1+(threadIdx.x-32)/16;
        const unsigned mask=0xffffu<<((threadIdx.x&31)>=16?16:0);
        pinv_group<3,16>(problem==1?position:orientation,
            problem==1?position_inverse:orientation_inverse,work[problem],
            ranks+problem,ok+problem,mask);
    }
}
