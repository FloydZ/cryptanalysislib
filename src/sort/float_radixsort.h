#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "traits.h"


#define CHECK_RESIZE(n)																			\
	if(n!=mPreviousSize)																		\
	{																							\
				if(n>mCurrentSize)	Resize(n);													\
		else						ResetIndices();												\
		mPreviousSize = n;																		\
	}

#define CREATE_HISTOGRAMS(type, buffer)															\
	/* Clear counters */																		\
	memset(mHistogram, 0, 256*4*sizeof(uint32_t));												\
																								\
	/* Prepare for temporal coherence */														\
	type PrevVal = (type)buffer[mIndices[0]];													\
	bool AlreadySorted = true;	/* Optimism... */												\
	uint32_t* Indices = mIndices;																\
																								\
	/* Prepare to count */																		\
	uint8_t* p = (uint8_t*)input;																\
	uint8_t* pe = &p[nb*4];																		\
	uint32_t* h0= &mHistogram[0];		/* Histogram for first pass (LSB)	*/					\
	uint32_t* h1= &mHistogram[256];	/* Histogram for second pass		*/						\
	uint32_t* h2= &mHistogram[512];	/* Histogram for third pass			*/						\
	uint32_t* h3= &mHistogram[768];	/* Histogram for last pass (MSB)	*/						\
																								\
	while(p!=pe)																				\
	{																							\
		/* Read input buffer in previous sorted order */										\
		type Val = (type)buffer[*Indices++];													\
		/* Check whether already sorted or not */												\
		if(Val<PrevVal)	{ AlreadySorted = false; break; } /* Early out */						\
		/* Update for next iteration */															\
		PrevVal = Val;																			\
																								\
		/* Create histograms */																	\
		h0[*p++]++;	h1[*p++]++;	h2[*p++]++;	h3[*p++]++;											\
	}																							\
																								\
	/* If all input values are already sorted, we just have to return and leave the */			\
	/* previous list unchanged. That way the routine may take advantage of temporal */			\
	/* coherence, for example when used to sort transparent faces.					*/			\
	if(AlreadySorted)	{ mNbHits++; return *this;	}											\
																								\
	/* Else there has been an early out and we must finish computing the histograms */			\
	while(p!=pe)																				\
	{																							\
		/* Create histograms without the previous overhead */									\
		h0[*p++]++;	h1[*p++]++;	h2[*p++]++;	h3[*p++]++;											\
	}

#define CHECK_PASS_VALIDITY(pass)																\
	/* Shortcut to current counters */															\
	uint32_t* CurCount = &mHistogram[pass<<8];													\
																								\
	/* Reset flag. The sorting pass is supposed to be performed. (default) */					\
	bool PerformPass = true;																	\
																								\
	/* Check pass validity */																	\
																								\
	/* If all values have the same byte, sorting is useless. */									\
	/* It may happen when sorting bytes or words instead of dwords. */							\
	/* This routine actually sorts words faster than dwords, and bytes */						\
	/* faster than words. Standard running time (O(4*n))is reduced to O(2*n) */					\
	/* for words and O(n) for bytes. Running time for floats depends on actual values... */		\
																								\
	/* Get first byte */																		\
	uint8_t UniqueVal = *(((uint8_t*)input)+pass);												\
																								\
	/* Check that byte's counter */																\
	if(CurCount[UniqueVal]==nb)	PerformPass=false;


class RadixSort : non_copyable {
public:
	// Constructor/Destructor
	constexpr RadixSort() noexcept {
        ResetIndices();
    }

	constexpr ~RadixSort() noexcept {
    }

	// Sorting methods
	constexpr RadixSort &Sort(const uint32_t *input,
                              const size_t nb) noexcept {
    	// Checkings
    	if(!input || !nb)	return *this;
    
    	// Stats
    	mTotalCalls++;
    
    	// Resize lists if needed
    	CHECK_RESIZE(nb);
    
    	// Allocate histograms & offsets on the stack
    	uint32_t mHistogram[256*4];
    	uint32_t mOffset[256];
    
    	// Create histograms (counters). Counters for all passes are created in one run.
    	// Pros:	read input buffer once instead of four times
    	// Cons:	mHistogram is 4Kb instead of 1Kb
    	// We must take care of signed/unsigned values for temporal coherence.... I just
    	// have 2 code paths even if just a single opcode changes. Self-modifying code, someone?
        CREATE_HISTOGRAMS(uint32_t, input);
    
    	// Radix sort, j is the pass number (0=LSB, 3=MSB)
    	for(uint32_t j=0;j<4;j++) {
    		CHECK_PASS_VALIDITY(j);
    
    		// Sometimes the fourth (negative) pass is skipped because all numbers are negative and the MSB is 0xFF (for example). This is
    		// not a problem, numbers are correctly sorted anyway.
    		if(PerformPass) {
    			// Should we care about negative values?
    			if(j!=3) {
    				// Here we deal with positive values only
    
    				// Create offsets
    				mOffset[0] = 0;
    				for(uint32_t i=1;i<256;i++)		mOffset[i] = mOffset[i-1] + CurCount[i-1];
    			} 
    
    			// Perform Radix Sort
    			uint8_t* InputBytes	= (uint8_t*)input;
    			uint32_t* Indices		= mIndices;
    			uint32_t* IndicesEnd	= &mIndices[nb];
    			InputBytes += j;
    			while(Indices!=IndicesEnd) {
    				uint32_t id = *Indices++;
    				mIndices2[mOffset[InputBytes[id<<2]]++] = id;
    			}
    
    			// Swap pointers for next pass. Valid indices - the most recent ones - are in mIndices after the swap.
    			uint32_t* Tmp	= mIndices;	mIndices = mIndices2; mIndices2 = Tmp;
    		}
    	}
    	return *this;
    }


	constexpr RadixSort &Sort(const float *input2, size_t nb) noexcept {
    	// Checkings
    	if(!input2 || !nb)	return *this;
    
    	// Stats
    	mTotalCalls++;
    
    	uint32_t* input = (uint32_t*)input2;
    
    	// Resize lists if needed
    	CHECK_RESIZE(nb);
    
    	// Allocate histograms & offsets on the stack
    	uint32_t mHistogram[256*4];
    	uint32_t mOffset[256];
    
    	// Create histograms (counters). Counters for all passes are created in one run.
    	// Pros:	read input buffer once instead of four times
    	// Cons:	mHistogram is 4Kb instead of 1Kb
    	// Floating-point values are always supposed to be signed values, so there's only one code path there.
    	// Please note the floating point comparison needed for temporal coherence! Although the resulting asm code
    	// is dreadful, this is surprisingly not such a performance hit - well, I suppose that's a big one on first
    	// generation Pentiums....We can't make comparison on integer representations because, as Chris said, it just
    	// wouldn't work with mixed positive/negative values....
    	{ CREATE_HISTOGRAMS(float, input2); }
    
    	// Compute #negative values involved if needed
    	uint32_t NbNegativeValues = 0;
    	// An efficient way to compute the number of negatives values we'll have to deal with is simply to sum the 128
    	// last values of the last histogram. Last histogram because that's the one for the Most Significant Byte,
    	// responsible for the sign. 128 last values because the 128 first ones are related to positive numbers.
    	uint32_t* h3= &mHistogram[768];
    	for(uint32_t i=128;i<256;i++)	NbNegativeValues += h3[i];	// 768 for last histogram, 128 for negative part
    
    	// Radix sort, j is the pass number (0=LSB, 3=MSB)
    	for(uint32_t j=0;j<4;j++) {
    		// Should we care about negative values?
    		if(j!=3) {
    			// Here we deal with positive values only
    			CHECK_PASS_VALIDITY(j);
    
    			if(PerformPass) {
    				// Create offsets
    				mOffset[0] = 0;
    				for(uint32_t i=1;i<256;i++)		mOffset[i] = mOffset[i-1] + CurCount[i-1];
    
    				// Perform Radix Sort
    				uint8_t* InputBytes	    = (uint8_t*)input;
    				uint32_t* Indices		= mIndices;
    				uint32_t* IndicesEnd	= &mIndices[nb];
    				InputBytes += j;
    				while(Indices!=IndicesEnd) {
    					uint32_t id = *Indices++;
    					mIndices2[mOffset[InputBytes[id<<2]]++] = id;
    				}
    
    				// Swap pointers for next pass. Valid indices - the most recent ones - are in mIndices after the swap.
    				uint32_t* Tmp	= mIndices;	mIndices = mIndices2; mIndices2 = Tmp;
    			}
    		} else {
    			// This is a special case to correctly handle negative values
    			CHECK_PASS_VALIDITY(j);
                uint32_t i; 
    			if(PerformPass) {
    				// Create biased offsets, in order for negative numbers to be sorted as well
    				mOffset[0] = NbNegativeValues;												// First positive number takes place after the negative ones
    				for(uint32_t i=1;i<128;i++)		mOffset[i] = mOffset[i-1] + CurCount[i-1];	// 1 to 128 for positive numbers
    
    				// We must reverse the sorting order for negative numbers!
    				mOffset[255] = 0;
    				for(i=0;i<127;i++)		mOffset[254-i] = mOffset[255-i] + CurCount[255-i];	// Fixing the wrong order for negative values
    				for(i=128;i<256;i++)	mOffset[i] += CurCount[i];							// Fixing the wrong place for negative values
    
    				// Perform Radix Sort
    				for(i=0;i<nb;i++) {
    					uint32_t Radix = input[mIndices[i]]>>24;								// Radix byte, same as above. AND is useless here (uint32_t).
    					// ### cmp to be killed. Not good. Later.
    					if(Radix<128)		mIndices2[mOffset[Radix]++] = mIndices[i];		// Number is positive, same as above
    					else				mIndices2[--mOffset[Radix]] = mIndices[i];		// Number is negative, flip the sorting order
    				}
    				// Swap pointers for next pass. Valid indices - the most recent ones - are in mIndices after the swap.
    				uint32_t* Tmp	= mIndices;	mIndices = mIndices2; mIndices2 = Tmp;
    			} else {
    				// The pass is useless, yet we still have to reverse the order of current list if all values are negative.
    				if(UniqueVal>=128) {
    					for(i=0;i<nb;i++)	mIndices2[i] = mIndices[nb-i-1];
    
    					// Swap pointers for next pass. Valid indices - the most recent ones - are in mIndices after the swap.
    					uint32_t* Tmp	= mIndices;	mIndices = mIndices2; mIndices2 = Tmp;
    				}
    			}
    		}
    	}
    	return *this;
    }


	//! Access to results. mIndices is a list of indices in sorted order, i.e. 
    // in the order you may further process your data
	constexpr inline uint32_t *GetIndices() const noexcept { 
        return mIndices; 
    }

	//! mIndices2 gets trashed on calling the sort routine, but otherwise you can recycle it the way you want.
	constexpr inline uint32_t *GetRecyclable() const noexcept { 
        return mIndices2; 
    }

	// Stats
	constexpr uint32_t GetUsedRam() const noexcept {
        // 2 lists of indices
    	uint32_t UsedRam = sizeof(RadixSort);
    	UsedRam += 2*mCurrentSize*sizeof(uint32_t);	
    	return UsedRam;
    }

	//! Returns the total number of calls to the radix sorter.
	constexpr inline uint32_t GetNbTotalCalls() const noexcept {
        return mTotalCalls; 
    }

	//! Returns the number of premature exits due to temporal coherence.
	constexpr inline uint32_t GetNbHits() const noexcept { 
        return mNbHits; 
    }

private:
	uint32_t mCurrentSize = 0; //!< Current size of the indices list
	uint32_t mPreviousSize = 0;//!< Size involved in previous call
	uint32_t *mIndices = nullptr;    //!< Two lists, swapped each pass
	uint32_t *mIndices2 = nullptr;
	// Stats
	uint32_t mTotalCalls = 0;
	uint32_t mNbHits = 0;

	// Internal methods
	bool Resize(const size_t nb) noexcept {
    	// Free previously used ram
        delete mIndices2;
        delete mIndices;

    	// Get some fresh one
    	mIndices		= new uint32_t[nb];
    	mIndices2		= new uint32_t[nb];
    	mCurrentSize	= nb;
    
    	// Initialize indices so that the input buffer is read in sequential order
    	ResetIndices();
    
    	return true;
    }

	constexpr void ResetIndices() noexcept {
    	for(uint32_t i=0;i<mCurrentSize;i++) {
            mIndices[i] = i; 
        }
    }
};
