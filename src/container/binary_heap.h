#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


#define SIZE 1000
typedef struct binaryHeap 
{
    void **data;
    uint64_t len;
    uint64_t used;
} binaryHeap;

inline void swap(void *e, void *b)
{
    void *a = e;
    e = b;
    b = a;
           
}

uint8_t binaryHeap_Init(binaryHeap *b, uint64_t size)
{
    b->len = size;
    if (size == 0)
        b->len = SIZE;
                
    b->data = (void *)malloc(sizeof(void *) * b->len);   
    if(b->data == NULL)
        return 1;
    
    b->used = 0;
    
    return 0;
}

uint8_t siftUp(binaryHeap *b, uint64_t j)
{
    //TODO assert h[j] ist event zu klein
    if((j == 1) || (b->data[j/2] <= b->data[j])) //TODO wie richtig auf werte referencieren // ab besten im init Callback func hinterlegen
        return 0;
    
    swap(b->data[j/2], b->data[j]);
    //TODO assert h[j/2] ist event zu klein           
    siftUp(b, j/2);
    return 0;
}

uint8_t insert(binaryHeap *b, void *e)
{
    if (b->used == b->len)
    {
        printf("TODO INSERT \n");
        return 1;
    }
    
    b->data[b->used++];
    siftUp(b, b->used);
    
    return 0;
}





uint8_t siftDown(binaryHeap *b, uint64_t j)
{
    //TODO assert h[j] ist event zu gross
    
    uint64_t m;
    if((2*j + 1 > b->used) || (b->data[2*j] <= b->data[2*j + 1])) //TODO Compare fkt
    {
        m = 2*j;
    }
    else
    {
        m = 2*j + 1;
    }
    
    //TODO Assert falls ein geschwisterknoten von m exisitiert ist er nicht groesser
    if (b->data[j] > b->data[m]) //todo richtig comparen
    {
        swap(b->data[j], b->data[m]);
        // TODO assert h[m] ist evtl zu gross
        siftDown(b, m);
                
    }
    return 0;
}


void* deleteMin(binaryHeap *b)
{
    void *result = b->data[0];
    b->data[0] = b->data[b->used];
    b->used--;
    siftDown(b, 1);
    
    return result;
}
