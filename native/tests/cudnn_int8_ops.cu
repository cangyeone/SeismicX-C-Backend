/* SPDX-License-Identifier: GPL-3.0-only */
/* Exercise production packing, phase decomposition and exact cuDNN integer sums.
 * Inputs are integral with max=127, making dynamic activation scale exactly one.
 * CPU oracle computes the original full transpose formula, not polyphase code. */
#include "../src/cudnn.cu"
#include <cstdint>
int main(int argc,char **argv){
    if(argc!=2)return 77;FILE *f=fopen(argv[1],"rb");if(!f)return 77;fclose(f);
    pnsn_options o=pnsn_default_options();o.window=256;o.overlap=64;
    pnsn_context *ctx=pnsn_open(argv[1],&o);if(!ctx){fprintf(stderr,"%s\n",pnsn_last_error());return 1;}
    DNN *d=(DNN*)ctx->gpu;if(!d->int8 || reserve(d,3)){pnsn_close(ctx);return 1;}
    float error=0;int integer_error=0,checked=0;
    for(int i=0;i<25;i++){
        Conv &c=d->conv[i];if(!c.op)continue;const pnsn_layer &l=ctx->layers[i];int t=c.t,u=c.u;
        std::vector<float>x(3*c.ni*t),y(3*c.no*u),acc(3*c.no*(c.transpose?t:u));std::vector<int32_t>sums(3*c.no*u,0);
        for(size_t j=0;j<x.size();j++)x[j]=(int)((j*37+13)%255)-127;
        for(int b=0;b<3;b++)for(int time=0;time<t;time++)x[(b*t+time)*c.ni]=127;
        if(check(cudaMemcpyAsync(d->a,x.data(),x.size()*4,cudaMemcpyHostToDevice,d->stream),__LINE__) || convolution(d,i,d->a,d->b,0) || check(cudaStreamSynchronize(d->stream),__LINE__) || check(cudaMemcpy(y.data(),d->b,y.size()*4,cudaMemcpyDeviceToHost),__LINE__)){pnsn_close(ctx);return 1;}
        for(int b=0;b<3;b++)for(int out=0;out<c.no;out++)for(int time=0;time<u;time++){
            int sum=0;
            for(int k=0;k<c.k;k++){
                int source=c.transpose?time+c.pad-k:time*(c.upsample?1:c.stride)+k-c.pad;
                if(c.upsample){if(source<0 || source>=t*4)continue;source/=4;}
                if(c.transpose){if(source%2)continue;source/=2;}if(source<0 || source>=t)continue;
                for(int in=0;in<c.ni;in++)sum+=(int)x[(b*t+source)*c.ni+in]*(int)l.q[out*l.width+k*c.ni+in];
            }
            int at=(b*u+time)*c.no+out;sums[at]=sum;float expected=(float)sum*l.scale[out]+l.bias[out];error=std::max(error,fabsf(expected-y[at]));
        }
        if(c.transpose){if(check(cudaMemcpy(acc.data(),d->accum,acc.size()*4,cudaMemcpyDeviceToHost),__LINE__)){pnsn_close(ctx);return 1;}}
        else {
            /* Re-run the same vendor integer op into an unscaled test buffer. */
            float *integer=nullptr,one=1,zero=0;IntConv &ic=c.integer[0];
            if(check(cudaMalloc(&integer,acc.size()*4),__LINE__) || check(cudnnConvolutionForward(d->handle,&one,ic.x,d->quant,ic.filter,ic.weight,ic.op,ic.algo,d->work,ic.workspace,&zero,ic.y,integer),__LINE__) || check(cudaStreamSynchronize(d->stream),__LINE__) || check(cudaMemcpy(acc.data(),integer,acc.size()*4,cudaMemcpyDeviceToHost),__LINE__)){cudaFree(integer);pnsn_close(ctx);return 1;}cudaFree(integer);
        }
        int last=c.transpose?c.integer[1].first:0;
        for(int b=0;b<3;b++)for(int time=0;time<(c.transpose?t:u);time++)for(int out=0;out<c.no;out++){
            int expected=sums[(b*u+(c.transpose?time*2+last:time))*c.no+out];
            integer_error=std::max(integer_error,abs((int)acc[(b*(c.transpose?t:u)+time)*c.no+out]-expected));
        }
        checked++;
    }
    pnsn_close(ctx);printf("{\"layers\":%d,\"batch\":3,\"int32_max_error\":%d,\"dequantized_max_error\":%.9g}\n",checked,integer_error,error);
    return integer_error==0 && error<1e-4?0:1;
}
