/* SPDX-License-Identifier: GPL-3.0-only */
#include "pnsn.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
/* Do not compile test assertions out in Release. */
#undef assert
#define assert(x) do { if(!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; } } while(0)
int main(void) {
    pnsn_pick input[]={{0,10,.6f},{0,11,.9f},{0,14,.8f},{1,11,.7f},{0,100,.8f},{0,103,.8f}},out[10];size_t n=0;
    assert(!pnsn_nms(input,6,3,out,10,&n));assert(n==3);
    assert(out[0].sample==11 && out[0].phase==0);assert(out[1].phase==1);assert(out[2].sample==100);
    out[0].sample=999;
    assert(pnsn_nms(input,6,3,out,1,&n)==-2 && n==3 && out[0].sample==999);
    assert(!pnsn_nms(NULL,0,0,NULL,0,&n) && !n);
    input[0].confidence=NAN;assert(pnsn_nms(input,6,3,out,10,&n)<0);
    pnsn_pick large[]={{0,INT64_MAX,.9f},{0,INT64_MAX-1,.8f}};
    assert(!pnsn_nms(large,2,INT64_MAX,out,10,&n) && n==1);
    pnsn_options o=pnsn_default_options();o.window=6143;
    assert(!pnsn_open("missing",&o));assert(!pnsn_open(NULL,NULL));
    puts(pnsn_cpu_features());return 0;
}
