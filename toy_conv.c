#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>

struct Input{
    int32_t *tIn;
    uint8_t *knl;
    int32_t *tOut;
    int tN1, tN2;
    int tIH, tIW, tIC, tOC;
    int tKH, tKW;
};

struct KNL{
  int32_t *knl_in;
  uint8_t *knl_col;
  int oc1, oc2, KH, KW, IC, OC;
};

double benchmark(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    int num_iter = 100;

    struct timespec start, end;
    double total_time = 0;

    for (int eval_iter=0;eval_iter<num_iter;eval_iter++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        inference(tensorIn, kernel, tensorOut, N, IH, IW, IC, OC, KH, KW);
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time += (double)(end.tv_sec - start.tv_sec)*1000000 + (double)(end.tv_nsec - start.tv_nsec)/1000.0;
    }

    return total_time / (double)(num_iter);
}

void conv(
    struct Input *in
)
{

  int32_t *tensorIn = in->tIn; uint8_t *kernel_col = in->knl;
  int32_t *tensorOut = in->tOut;
  int IH = in->tIH, IW = in->tIW, IC = in->tIC, OC = in->tOC;
  int KH = in->tKH, KW = in->tKW;

  int KH2 = KH/2, KW2 = KW/2;
  int IHIWIC = IH*IW*IC, IWIC = IW*IC, KHKWIC = KH*KW*IC, KWIC = KW*IC, IHIWOC = IH*IW*OC, IWOC = IW*OC, KWKH = KW*KH;


  uint8_t *input_col = malloc(sizeof(uint8_t) * (IHIWIC * KWKH));
  for(int n = in->tN1; n < in->tN2; ++n){
    for(int h=-KH2; h < IH - KH2; ++h){
      for(int w = -KW2 ; w < IW - KW2; ++w){
          int tensor_s = w*IC + h*(IWIC) + n*(IHIWIC);
          int col_s = (w+KW2) + (h+KH2)*IW;
  			  for(int ic = 0; ic < IC; ++ic){
    			  for(int kh=0; kh<KH; ++kh){
    		      for(int kw=0; kw<KW; ++kw){
  						  if ((h+kh>=0) && (h+kh<IH) && (w+kw>=0) && (w+kw<IW))
                  input_col[col_s * (KHKWIC) + ic*(KWKH) + kh*(KW) + kw] = tensorIn[tensor_s + ic + kw*(IC) + kh*(IWIC)];
                else
                  input_col[col_s * (KHKWIC) + ic*(KWKH) + kh*(KW) + kw] = 0;
  			  }}}
    }}
    for(int h = 0; h < IH; ++h){
      for(int w = 0; w < IW; ++w){
        int input_s = h*IW + w;
        int output_s = (n)*(IHIWOC) + (h)*(IWOC) + (w)*(OC);
        for(int oc = 0; oc < OC; oc++){
          int32_t val = 0;
          for(int i = 0; i < KHKWIC; i++)
            val += input_col[input_s * (KHKWIC) + i] * kernel_col[oc * (KHKWIC) + i];
          tensorOut[output_s + oc] = val;
    }}}
   }
  free(input_col);
}

void knl_func (
  struct KNL *in
){
  int KWKHIC = in->KH*in->KW*in->IC, KWIC = in->KW*in->IC, KWKH = in->KW*in->KH;

 uint8_t *kernel_col = in->knl_col;
  int32_t *kernel = in->knl_in;
  for(int oc = in->oc1; oc < in->oc2; ++oc){
    for(int ic = 0; ic < in->IC; ++ic){
      for(int kh = 0; kh < in->KH; ++kh){
        for(int kw = 0; kw < in->KW; ++kw){
          kernel_col[kw + kh*(in->KW) + ic*(KWKH) + oc*(KWKHIC)] = kernel[ic + kw*(in->IC) + kh*(KWIC) + oc*(KWKHIC)];
  }}}}
}

  
int inference(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    /* Code Starts Here */
  pthread_t pthread[4];
  int thr_id[4], thN=4;
  int IWIHOC = IW*IH*OC, KWKH = KW*KH;
  int KWIC = KW*IC, KWKHIC = KWKH*IC;

  uint8_t *kernel_col = malloc(sizeof(uint8_t) * (OC*KWKHIC));

  struct KNL *pi[4];
  for (int i=0;i<thN;i++)
    pi[i] = malloc(sizeof(struct Input));

  for(int i=0;i<thN;i++){
    pi[i]->knl_in = kernel; pi[i]->knl_col = kernel_col;
    pi[i]->KH = KH; pi[i]->KW = KW; pi[i]->IC = IC; pi[i]->OC;
    pi[i]->oc1 = i*OC/thN; pi[i]->oc2 = (i+1)*OC/thN;

    thr_id[i] = pthread_create(&pthread[i], NULL, knl_func, (void*)(pi[i]));
  }
  for (int i=0;i<thN;i++)
    pthread_join(pthread[i], NULL);
  
  for (int i=0;i<thN;i++)
    free(pi[i]);


  struct Input *p[4];
  for (int i=0;i<thN;i++)
    p[i] = malloc(sizeof(struct Input));

  for(int i=0;i<thN;i++){
    p[i]->tIn = tensorIn; p[i]->knl = kernel_col; p[i]->tOut = tensorOut; 
    p[i]->tIH = IH; p[i]->tIW = IW; p[i]->tIC = IC; p[i]->tOC = OC;
    p[i]->tKH = KH; p[i]->tKW = KW;
    p[i]->tN1 = i*N/thN; p[i]->tN2 = (i+1)*N/thN;

    thr_id[i] = pthread_create(&pthread[i], NULL, conv, (void*)(p[i]));
  }
  for (int i=0;i<thN;i++)
    pthread_join(pthread[i], NULL);

  for (int i=0;i<thN;i++)
    free(p[i]);

  return 0;
    /* Code Ends Here */
}
