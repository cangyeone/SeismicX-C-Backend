/* SPDX-License-Identifier: GPL-3.0-only */
#include "internal.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
static int coordinate_order(const void *a,const void *b) {
    const pnsn_pick *x=a,*y=b;
    if(x->phase!=y->phase) return x->phase<y->phase?-1:1;
    if(x->sample!=y->sample) return x->sample<y->sample?-1:1;
    return x->confidence>y->confidence?-1:x->confidence<y->confidence;
}
static int time_order(const void *a,const void *b) {
    const pnsn_pick *x=a,*y=b;
    if(x->sample!=y->sample) return x->sample<y->sample?-1:1;
    return x->phase<y->phase?-1:x->phase>y->phase;
}
/* Smaller key = higher confidence. An actual binary min-heap, no top-K pruning. */
static int less(size_t a,size_t b,const pnsn_pick *p) {
    if(p[a].confidence!=p[b].confidence) return p[a].confidence>p[b].confidence;
    if(p[a].sample!=p[b].sample) return p[a].sample<p[b].sample;
    if(p[a].phase!=p[b].phase) return p[a].phase<p[b].phase;
    return a<b;
}
static void sift(size_t *heap,size_t n,size_t at,const pnsn_pick *p) {
    for(;;) {
        size_t child=2*at+1;if(child>=n)return;
        if(child+1<n && less(heap[child+1],heap[child],p))child++;
        if(!less(heap[child],heap[at],p))return;
        size_t v=heap[at];heap[at]=heap[child];heap[child]=v;at=child;
    }
}
static size_t successor(size_t *next,size_t i) {
    size_t root=i;while(next[root]!=root)root=next[root];
    while(next[i]!=i){size_t j=next[i];next[i]=root;i=j;}return root;
}
static size_t lower(const pnsn_pick *p,size_t n,int phase,int64_t sample,int upper) {
    size_t lo=0,hi=n;
    while(lo<hi) {size_t mid=lo+(hi-lo)/2;
        if(p[mid].phase<phase || (p[mid].phase==phase && (p[mid].sample<sample || (upper && p[mid].sample==sample)))) lo=mid+1;else hi=mid;
    }return lo;
}
int pnsn_nms(const pnsn_pick *input,size_t n,int64_t gap,pnsn_pick *out,size_t cap,size_t *count) {
    if(!count || (n && !input) || (cap && !out) || gap<0 || n>SIZE_MAX/sizeof(pnsn_pick) || n>=SIZE_MAX/sizeof(size_t)) return pnsn_error("Invalid NMS arguments");
    *count=0;if(!n)return 0;
    for(size_t i=0;i<n;i++) if(input[i].phase<0 || input[i].phase>3 || input[i].sample<0 || !isfinite(input[i].confidence) || input[i].confidence<0 || input[i].confidence>1) return pnsn_error("Invalid NMS candidate");
    pnsn_pick *p=malloc(n*sizeof(*p)),*selected=malloc(n*sizeof(*selected));
    size_t *heap=malloc(n*sizeof(*heap)),*next=malloc((n+1)*sizeof(*next));
    if(!p || !selected || !heap || !next) {free(p);free(selected);free(heap);free(next);return pnsn_error("Out of memory");}
    memcpy(p,input,n*sizeof(*p));qsort(p,n,sizeof(*p),coordinate_order);
    for(size_t i=0;i<n;i++){heap[i]=i;next[i]=i;}next[n]=n;
    for(size_t i=n/2;i>0;i--)sift(heap,n,i-1,p);
    size_t remaining=n,used=0;
    while(remaining) {
        size_t index=heap[0];heap[0]=heap[--remaining];if(remaining)sift(heap,remaining,0,p);
        if(successor(next,index)!=index)continue;
        pnsn_pick pick=p[index];selected[used++]=pick;
        int64_t low=pick.sample<gap?0:pick.sample-gap;
        int64_t high=pick.sample>INT64_MAX-gap?INT64_MAX:pick.sample+gap;
        size_t begin=lower(p,n,pick.phase,low,0),end=lower(p,n,pick.phase,high,1);
        for(size_t i=successor(next,begin);i<end;i=successor(next,i))next[i]=successor(next,i+1);
    }
    *count=used;int status=0;
    if(used>cap) {pnsn_error("Pick output capacity insufficient; count contains required capacity");status=-2;}
    else {qsort(selected,used,sizeof(*selected),time_order);if(used)memcpy(out,selected,used*sizeof(*out));}
    free(p);free(selected);free(heap);free(next);return status;
}
