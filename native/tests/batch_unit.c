#include "pnsn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL %s:%d %s: %s\n",__FILE__,__LINE__,#x,pnsn_last_error());return 1; } } while(0)
int main(int argc,char **argv) {
    if(argc!=2)return 77;
    FILE *f=fopen(argv[1],"rb");if(!f)return 77;fclose(f);
    pnsn_options o=pnsn_default_options();o.window=256;o.overlap=64;o.threshold=0;o.min_gap=4;
    pnsn_context *c=pnsn_open(argv[1],&o);CHECK(c);int classes=pnsn_classes(c);
    int32_t input[3*3*256],interleaved[3*256];float out[3*5*256],reference[5*256];
    for(int i=0;i<3*3*256;i++)input[i]=(i*37)%103-51;
    CHECK(!pnsn_predict_batch(c,input,PNSN_I32,3,256,out));
    for(int b=0;b<3;b++) {
        for(int t=0;t<256;t++)for(int channel=0;channel<3;channel++)interleaved[t*3+channel]=input[b*3*256+channel*256+t];
        CHECK(!pnsn_predict(c,interleaved,PNSN_I32,256,reference));
        CHECK(!memcmp(reference,out+b*classes*256,256*classes*sizeof(float)));
    }
    pnsn_pick selected[3*1024],history[1024];size_t offsets[4],count=0;
    CHECK(!pnsn_pick_batch(c,input,PNSN_I32,3,256,selected,3*1024,offsets,&count));CHECK(offsets[0]==0 && offsets[3]==count);
    for(int b=0;b<3;b++) {
        size_t n=0;CHECK(!pnsn_pick_history(c,input+b*3*256,PNSN_I32,256,history,1024,&n));CHECK(n==offsets[b+1]-offsets[b]);
        for(size_t j=0;j<n;j++) {pnsn_pick *p=selected+offsets[b]+j;CHECK(history[j].phase==p->phase && history[j].sample==p->sample && history[j].confidence==p->confidence);}
    }
    selected[0].sample=777;for(int i=0;i<4;i++)offsets[i]=777;
    CHECK(pnsn_pick_batch(c,input,PNSN_I32,3,256,selected,1,offsets,&count)==-2);CHECK(selected[0].sample==777 && offsets[0]==777 && count>1);
    CHECK(!pnsn_predict_batch(c,NULL,PNSN_I32,0,256,NULL));
    CHECK(!pnsn_pick_batch(c,NULL,PNSN_I32,0,256,NULL,0,offsets,&count));CHECK(count==0 && offsets[0]==0);
    CHECK(pnsn_predict_batch(c,input,PNSN_I32,3,127,out)<0);
    pnsn_close(c);puts("Planar history and independent realtime batch C API passed");return 0;
}
