/*
**  $Id: lzmat_enc.c,v 1.1 2008/07/08 16:58:35 Vitaly Exp $
**  $Revision: 1.1 $
**  $Date: 2008/07/08 16:58:35 $
** 
**  $Author: Vitaly $
**
***************************************************************************
** LZMAT ANSI-C encoder 1.01
** Copyright (C) 2007,2008 Vitaly Evseenko. All Rights Reserved.
** lzmat_enc.c
**
** This file is part of the LZMAT real-time data compression library.
**
** The LZMAT library is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License as
** published by the Free Software Foundation; either version 2 of
** the License, or (at your option) any later version.
**
** The LZMAT library is distributed WITHOUT ANY WARRANTY;
** without even the implied warranty of MERCHANTABILITY or
** FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
** License for more details.
**
** You should have received a copy of the GNU General Public License
** along with the LZMAT library; see the file GPL.TXT.
** If not, write to the Free Software Foundation, Inc.,
** 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
**
** Vitaly Evseenko
** <ve@matcode.com>
** http://www.matcode.com/lzmat.htm
***************************************************************************
*/


#ifndef CRYPTANALYSISLIB_COMPRESSION_LZMAT_H
#define CRYPTANALYSISLIB_COMPRESSION_LZMAT_H

#include <memory.h>
#include <stdint.h>

#define GET_LE64(_p_) (*((uint64_t *) (_p_)))
#define GET_LE32(_p_) (*((uint32_t *) (_p_)))
#define GET_LE16(_p_) (*((uint16_t *) (_p_)))

#define SET_LE64(_p_, _v_) (*((uint64_t *) (_p_)) = (_v_))
#define SET_LE32(_p_, _v_) (*((uint32_t *) (_p_)) = (_v_))
#define SET_LE16(_p_, _v_) (*((uint16_t *) (_p_)) = (_v_))

#define MAX_LZMAT_ENCODED_SIZE(_sz_)	((_sz_)+(((_sz_)+7)>>3)+0x21)
#define LZMAT_STATUS_OK	0
#define LZMAT_STATUS_ERROR	(-1)
#define LZMAT_STATUS_INTEGRITY_FAILURE	0x100
#define LZMAT_STATUS_BUFFER_TOO_SMALL	0x110


#define LZMAT_DEFAULT_CNT	(0x12)
#define LZMAT_1BYTE_CNT		(0xFF + LZMAT_DEFAULT_CNT)
#define LZMAT_2BYTE_CNT		(0xFFFF + LZMAT_1BYTE_CNT)
#define LZMAT_MAX_2BYTE_CNT	(LZMAT_2BYTE_CNT-1)

#define LZMAT_GET_U4(_p_,_i_,_n_) \
	((_n_^=1)?(((uint8_t *)(_p_))[_i_]&0xF):(((uint8_t *)(_p_))[_i_++]>>4))

#define LZMAT_GET_U8(_p_,_n_)	(uint8_t)(((_n_)?((((uint8_t *)(_p_))[0]>>4)|(((uint8_t *)(_p_))[1]<<4)):((uint8_t *)(_p_))[0]))

#define LZMAT_GET_LE16(_p_,_n_) \
 (uint16_t)((_n_)?((((uint8_t *)(_p_))[0]>>4)|((uint16_t)(GET_LE16((_p_)+1))<<4)):GET_LE16(_p_))

#define MAX_LZMAT_SHORT_DIST0	0x80
#define MAX_LZMAT_SHORT_DIST1	(0x800|MAX_LZMAT_SHORT_DIST0)
#define MAX_LZMAT_LONG_DIST0	0x40
#define MAX_LZMAT_LONG_DIST1	(0x400|MAX_LZMAT_LONG_DIST0)
#define MAX_LZMAT_LONG_DIST2	(0x4000|MAX_LZMAT_LONG_DIST1)
#define MAX_LZMAT_LONG_DIST3	(0x40000|MAX_LZMAT_LONG_DIST2)
#define MAX_LZMAT_GAMMA_DIST	(MAX_LZMAT_LONG_DIST3-1)

#define LZMAT_DIST_MSK0		0x3F
#define LZMAT_DIST_MSK1		0x3FF



#define CB_LZMAT_IDX		0x20000
#define CB_LZMAT_DICT	0x20000



#ifdef LZMAT_SMALL_STACK
#define LZMAT_DICTIONATY_SIZE	((CB_LZMAT_IDX+CB_LZMAT_DICT))
uint32_t lzmat_dictionary_size(void) { return LZMAT_DICTIONATY_SIZE; }
#endif


#define RS_HASH_BITS	9


#define MAX_LZMAT_DICT	(CB_LZMAT_DICT/sizeof(uint32_t))
#define MAX_LZMAT_IDX	(CB_LZMAT_IDX/sizeof(uint32_t))


#define DICT_MSK	(MAX_LZMAT_DICT-1)
#define IDX_MSK	(MAX_LZMAT_IDX-1)

#define LZMAT_HASH(p) \
	((*(uint16_t *)(p)+(uint16_t)((*(uint32_t *)(p))>>RS_HASH_BITS))&DICT_MSK)

typedef struct _LZMAT_HASH_CTL {
	int32_t ptr[MAX_LZMAT_DICT];
	int32_t idx[MAX_LZMAT_IDX];
} LZMAT_HASH_CTL, *PLZMAT_HASH_CTL;

#define LZMAT_DEFAULT_CNT		(0x12)
#define LZMAT_1BYTE_CNT		(0xFF + LZMAT_DEFAULT_CNT)
#define LZMAT_2BYTE_CNT		(0xFFFF + LZMAT_1BYTE_CNT)
#define LZMAT_MAX_2BYTE_CNT	(LZMAT_2BYTE_CNT-1)

#define MAX_LZMAT_SHORT_DIST0	0x80
#define MAX_LZMAT_SHORT_DIST1	(0x800|MAX_LZMAT_SHORT_DIST0)
#define MAX_LZMAT_LONG_DIST0	0x40
#define MAX_LZMAT_LONG_DIST1	(0x400|MAX_LZMAT_LONG_DIST0)
#define MAX_LZMAT_LONG_DIST2	(0x4000|MAX_LZMAT_LONG_DIST1)
#define MAX_LZMAT_LONG_DIST3	(0x40000|MAX_LZMAT_LONG_DIST2)
#define MAX_LZMAT_GAMMA_DIST	(MAX_LZMAT_LONG_DIST3-1)

#ifdef LZMAT_SMALL_STACK
#define LZMATHDICT	(((PLZMAT_HASH_CTL)pVocabularyBuf)[0])
#else
#define LZMATHDICT	(lzh)
#endif


#define MAX_LZMAT_UNCOMP_BLOCK	0x280

#define MP_CMP_DISTANCE(_ptr_,_d1_,_d2_,_c1_,_c2_) \
	(_c2_>_c1_ && (_c2_>(_c1_+1) || (_ptr_>0x880 && (_d1_<<7)>_d2_) || ((_d1_<<3)>_d2_)))


#define LZMAT_GET_U4(_p_,_i_,_n_) \
	((_n_^=1)?(((uint8_t *)(_p_))[_i_]&0xF):(((uint8_t *)(_p_))[_i_++]>>4))

#define LZMAT_GET_U8(_p_,_n_)	(uint8_t)(((_n_)?((((uint8_t *)(_p_))[0]>>4)|(((uint8_t *)(_p_))[1]<<4)):((uint8_t *)(_p_))[0]))

#define LZMAT_GET_LE12(_p_,_n_) \
 ((_n_^=1)?((((uint16_t)((uint8_t *)(_p_))[1]&0xF)<<8)|((uint8_t *)(_p_))[0]):((((uint8_t *)(_p_))[0]>>4)|((uint16_t)((uint8_t *)(_p_))[1]<<4)))

#define LZMAT_GET_LE16(_p_,_n_) \
 (uint16_t)((_n_)?((((uint8_t *)(_p_))[0]>>4)|((uint16_t)(GET_LE16((_p_)+1))<<4)):GET_LE16(_p_))


#define LZMAT_SET_U4(_p_,_n_,_v_) { \
	if(_n_^=1) *(_p_) = (uint8_t)((_v_)&0xF); \
	else  *(_p_)++ |= (uint8_t)(_v_<<4);  }

#define LZMAT_SET_U8(_p_,_n_,_v_) { \
	if(_n_) { \
		*(_p_)++ |= (uint8_t)(_v_<<4); \
		 *(_p_) = (uint8_t)((_v_)>>4); \
	} else \
		*(_p_)++ = (uint8_t)(_v_); \
	}

#define LZMAT_SET_LE12(_p_,_n_,_v_) { \
	if(_n_^=1) { \
		*(_p_)++ = (uint8_t)(_v_); \
		*(_p_) = (uint8_t)((_v_)>>8); \
	} else { \
		*(_p_)++ |= (uint8_t)(_v_<<4); \
		*(_p_)++ = (uint8_t)((_v_)>>4); \
	} }

#define LZMAT_SET_LE16(_p_,_n_,_v_) { \
	if(_n_) { \
		*(_p_)++ |= (uint8_t)(_v_<<4); \
		*(_p_)++ = (uint8_t)((_v_)>>4); \
		*(_p_) = (uint8_t)((_v_)>>12); \
	} else { \
		SET_LE16(_p_,(uint16_t)(_v_)); (_p_)+=2; \
	} }

#define LZMAT_SET_LE20(_p_,_n_,_v_) { \
	if(_n_^=1) { \
		SET_LE16(_p_,(uint16_t)(_v_)); (_p_)+=2; \
		*(_p_) = (uint8_t)((_v_)>>16); \
	} else { \
		*(_p_)++ |= (uint8_t)((_v_)<<4); \
		SET_LE16(_p_,((uint16_t)((_v_)>>4))); (_p_)+=2; \
	} }




#define LZMAT_GET_LE12_UNSAVE(_p_,_n_) \
 ((_n_)?((GET_LE16(_p_)>>4)&0xFFF):(GET_LE16(_p_)&0xFFF))

#define LZMAT_GET_LE16_UNSAVE(_p_,_n_) \
 ((_n_)?((GET_LE32(_p_)>>4)&0xFFFF):GET_LE16(_p_))

#define LZMAT_GET_LE20_UNSAVE(_p_,_n_) \
 ((_n_)?((GET_LE32(_p_)>>4)&0xFFFFF):(GET_LE32(_p_)&0xFFFFF))






int lzmat_encode(uint8_t *pbOut, uint32_t *pcbOut,
                                uint8_t *pbIn, uint32_t cbIn
#ifdef LZMAT_SMALL_STACK
                                , void *pVocabularyBuf
#endif
)
{
	uint32_t i, match_cnt, inPtr, cpy_tag, cbUCData;
	uint8_t *pOut, *pTag, *pEndOut, *pInp;
	uint32_t Gamma_dist;
	uint8_t bit_msk, ThisTag, cur_nib, tag_nib, uc_nib;
	uint8_t *pUC_Tag;
	uint32_t processed_data;

#ifndef LZMAT_SMALL_STACK
	LZMAT_HASH_CTL lzh;
#endif

	uc_nib = cur_nib = tag_nib = 0;
	memset(&LZMATHDICT, -1, sizeof(LZMAT_HASH_CTL));
	pTag = pbOut+1;
	pUC_Tag = pTag;

	Gamma_dist = 0;
	bit_msk = 0x80;
	ThisTag = 0x00;

	pEndOut = pbOut + *pcbOut - 0x21;

	pOut = pTag+1;
	*pbOut = *pbIn;

	cpy_tag = 0;

	cbUCData = 0;
	processed_data = inPtr = 1;

	LZMATHDICT.ptr[LZMAT_HASH(pbIn)] = 0;

	while(cbIn > inPtr)
	{
		uint8_t *pITmp;
		uint32_t store_dist;
		uint16_t hash_Idx;

		pInp = pbIn + inPtr;
		pITmp = pInp - 1;
		if( (cbIn-inPtr)>=4 && *(uint32_t *)pInp == *(uint32_t *)pITmp )
		{
			uint32_t in_Reminder = cbIn-4-inPtr;
			uint8_t *pCurPtr = pInp+4;
			Gamma_dist = 0;
			pITmp+=4;
			match_cnt = 4;
			if(in_Reminder>(LZMAT_MAX_2BYTE_CNT-4))
				in_Reminder = (LZMAT_MAX_2BYTE_CNT-4);
			while(in_Reminder-- && *pCurPtr++ == *pITmp++)
				match_cnt++;
		}
		else
		{
			int32_t dict_ptr, cur_idx, start_pos;
			uint16_t cmp_val;
			match_cnt = 1;
			if ( (unsigned int)inPtr < MAX_LZMAT_GAMMA_DIST)
				start_pos = 0;
			else
				start_pos = inPtr - MAX_LZMAT_GAMMA_DIST;
			dict_ptr = LZMATHDICT.ptr[LZMAT_HASH(pInp)];
			if ( dict_ptr < start_pos)
				goto skip_search;
			cmp_val = *(uint16_t *)(pbIn + inPtr + 1);
			pITmp = pbIn + 1;
			cur_idx = inPtr;
			while ( dict_ptr < cur_idx )
			{
				cur_idx = dict_ptr;
				if ( *(uint16_t *)(pITmp + dict_ptr) == cmp_val )
				{
					uint32_t in_Reminder, new_dist, match_found=0;
					uint8_t *pIdxPtr, *pCurPtr;
					pIdxPtr = pbIn + dict_ptr; // + sizeof(uint16_t);
					pCurPtr = pInp; // + sizeof(uint16_t);
					in_Reminder = cbIn-inPtr;
					if(in_Reminder>LZMAT_MAX_2BYTE_CNT)
						in_Reminder = LZMAT_MAX_2BYTE_CNT;
					while(in_Reminder-- && *pIdxPtr++ == *pCurPtr++)
						match_found++;
					new_dist = inPtr - dict_ptr - 1;
					if(MP_CMP_DISTANCE(inPtr,Gamma_dist,new_dist,match_cnt,match_found))
					{
						Gamma_dist = new_dist;
						match_cnt = match_found;
						cmp_val = *(uint16_t *)(match_found + pInp - 1);
						pITmp = pbIn + match_found - 1;
						if (match_found >= LZMAT_MAX_2BYTE_CNT)
						{
							match_found = LZMAT_MAX_2BYTE_CNT;
							break;
						}
					}
				}
				dict_ptr = LZMATHDICT.idx[dict_ptr & IDX_MSK];
				if (dict_ptr < start_pos)
					break;
			}
		}
	skip_search:
		if(match_cnt > (cbIn-inPtr))
			match_cnt = cbIn-inPtr;

		if(match_cnt<3)
		{
			match_cnt = 1;
			LZMAT_SET_U8(pOut, cur_nib, pInp[0]);
			goto set_next_tag;
		}

		if(inPtr>MAX_LZMAT_SHORT_DIST1)
		{
			store_dist = Gamma_dist<<2;
			if(Gamma_dist<MAX_LZMAT_LONG_DIST0)
			{
				LZMAT_SET_U8(pOut,cur_nib,store_dist);
			}
			else if(Gamma_dist<MAX_LZMAT_LONG_DIST1)
			{
				store_dist-=(MAX_LZMAT_LONG_DIST0<<2);
				store_dist|= 1;
				LZMAT_SET_LE12(pOut,cur_nib,store_dist);
			}
			else if(Gamma_dist<MAX_LZMAT_LONG_DIST2)
			{
				store_dist-=(MAX_LZMAT_LONG_DIST1<<2);
				store_dist|= 2;
				LZMAT_SET_LE16(pOut,cur_nib,store_dist);
			}
			else
			{
				if(match_cnt<4)
				{
					match_cnt = 1;
					LZMAT_SET_U8(pOut, cur_nib, pInp[0]);
					goto set_next_tag;
				}
				store_dist-=(MAX_LZMAT_LONG_DIST2<<2);
				store_dist|= 3;
				LZMAT_SET_LE20(pOut,cur_nib,store_dist);
			}
		}
		else // short distance
		{
			store_dist = Gamma_dist<<1;
			if(Gamma_dist>=MAX_LZMAT_SHORT_DIST0)
			{
				store_dist-=(MAX_LZMAT_SHORT_DIST0<<1);
				store_dist|=1;
				LZMAT_SET_LE12(pOut,cur_nib,store_dist);
			}
			else
				LZMAT_SET_U8(pOut,cur_nib,store_dist);
		}
#define Stored_Cnt	Gamma_dist
		if(match_cnt<LZMAT_DEFAULT_CNT)
		{
			Stored_Cnt = match_cnt-3;
			LZMAT_SET_U4(pOut,cur_nib, Stored_Cnt);
		}
		else if(match_cnt<LZMAT_1BYTE_CNT)
		{
			Stored_Cnt = ((match_cnt-0x12)<<4)|0xF;
			LZMAT_SET_LE12(pOut,cur_nib,Stored_Cnt);
		}
		else
		{
			LZMAT_SET_LE12(pOut,cur_nib, 0xFFF);
			Stored_Cnt = match_cnt-0x111;
			LZMAT_SET_LE16(pOut,cur_nib, Stored_Cnt);
		}

#undef Stored_Cnt

#define cbCompressed	Gamma_dist
		ThisTag |= bit_msk;
	set_next_tag:
		bit_msk >>= 1;
		if ( bit_msk == 0 )
		{
			if(cpy_tag && cbUCData>0xFFF8)
			{
				uint32_t *pdwIn, *pdwOut;
				uint32_t cbCopy;
			copy_uncmp:
				cbCopy = (uint16_t)(cbUCData>>3);
				cbCompressed = cbCopy-4;
				cbCompressed = (cbCompressed&0xFF)|0x80|((cbCompressed<<3)&0xFC00);
				pUC_Tag[0]&=0xF; // !!! zero upper nibble in case cur_nib
				LZMAT_SET_LE16(pUC_Tag,uc_nib, cbCompressed);
				if(uc_nib)
				{
					LZMAT_SET_LE12(pUC_Tag, uc_nib, 0xFFF);
				}
				else
				{
					LZMAT_SET_LE16(pUC_Tag, uc_nib, 0xFFFF);
				}
				LZMAT_SET_LE16(pUC_Tag, uc_nib, 0xFFFF);
				pdwOut = (uint32_t *)pUC_Tag;
				cpy_tag = cbCopy<<1;
				pdwIn = (uint32_t *)(pbIn + processed_data);
				inPtr = processed_data+(cpy_tag<<2);
				while(cpy_tag--)
					*pdwOut++ = *pdwIn++;
				pUC_Tag = pTag = (uint8_t *)pdwOut;
				pOut = pTag+1;
				match_cnt = 1;
				cpy_tag =  0;
				tag_nib = cur_nib = 0;
				processed_data = inPtr;
				bit_msk = 0x80;
				ThisTag = 0;
				pInp = pbIn + inPtr;
				cbUCData = 0;
				continue;
			}
			i=inPtr-(processed_data+match_cnt);
			cbCompressed = (uint32_t)(pOut-pUC_Tag);
			if(ThisTag)
			{
				if(cpy_tag>0xff)
					goto copy_uncmp;
				else if(cbCompressed<i)
				{
					if(cpy_tag>0x3F)
						goto copy_uncmp;
					cpy_tag = 0;
					pUC_Tag = pOut;
					uc_nib = cur_nib;
					processed_data = inPtr+match_cnt;
				}
			}
			else
			{
				if(cpy_tag || (i+4)<cbCompressed)
				{
					cbUCData = inPtr - processed_data;
					cpy_tag++;
				}
			}
			if(tag_nib&1)
			{
				*pTag++ |= (ThisTag<<4);
				pTag[0] |= (ThisTag>>4);
			}
			else
				*pTag = ThisTag;
			bit_msk = 0x80;
			ThisTag = 0;
			pTag = pOut++;
			if(cur_nib)
				*pOut=0;
			tag_nib = cur_nib;
			if ( (uintptr_t)pOut >= (uintptr_t)pEndOut )
				return LZMAT_STATUS_INTEGRITY_FAILURE;
		}
#undef cbCompressed
		if(match_cnt==1)
		{
			hash_Idx = LZMAT_HASH(pInp);
			LZMATHDICT.idx[inPtr & IDX_MSK] = LZMATHDICT.ptr[hash_Idx];
			LZMATHDICT.ptr[hash_Idx] = inPtr++;
		}
		else
		{
			i = inPtr;
			inPtr += match_cnt;
			if ( (unsigned int)match_cnt > 0x38 )
				match_cnt = 0x38;
			while ( match_cnt-- )
			{
				hash_Idx = LZMAT_HASH(pInp);
				pInp++;
				LZMATHDICT.idx[i & IDX_MSK] = LZMATHDICT.ptr[hash_Idx];
				LZMATHDICT.ptr[hash_Idx] = (uint32_t)(i++); //((i++)-start_dict);
			}
		}
	}
	if(bit_msk)
	{
		ThisTag |= bit_msk;
		ThisTag |= (bit_msk-1);
	}
	if(tag_nib&1)
	{
		*pTag++ |= (ThisTag<<4);
		pTag[0] |= (ThisTag>>4);
	}
	else
		*pTag=ThisTag;
	*pcbOut = ((uintptr_t)pOut - (uintptr_t)pbOut)+cur_nib;
	return 0;
}
int lzmat_decode(uint8_t *pbOut, uint32_t *pcbOut,
                                uint8_t *pbIn, uint32_t cbIn)
{
	uint32_t  inPos, outPos;
	uint32_t  cbOutBuf = *pcbOut;
	uint8_t cur_nib;
	*pbOut = *pbIn;
	for(inPos=1, outPos=1, cur_nib=0; inPos<(cbIn-cur_nib);)
	{
		int bc;
		uint8_t tag;
		tag = LZMAT_GET_U8(pbIn+inPos,cur_nib);
		inPos++;
		for(bc=0; bc<8 && inPos<(cbIn-cur_nib) && outPos<cbOutBuf; bc++, tag<<=1)
		{
			if(tag&0x80) // gamma
			{
				uint32_t r_pos, r_cnt, dist;
#define cflag	r_cnt
				cflag = LZMAT_GET_LE16(pbIn+inPos,cur_nib);
				inPos++;
				if(outPos>MAX_LZMAT_SHORT_DIST1)
				{
					dist = cflag>>2;
					switch(cflag&3)
					{
						case 0:
							dist=(dist&LZMAT_DIST_MSK0)+1;
							break;
						case 1:
							inPos+=cur_nib;
							dist = (dist&LZMAT_DIST_MSK1)+0x41;
							cur_nib^=1;
							break;
						case 2:
							inPos++;
							dist += 0x441;
							break;
						case 3:
							if((inPos+2+cur_nib)>cbIn)
								return LZMAT_STATUS_INTEGRITY_FAILURE+1;
							inPos++;
							dist = (dist +
							        ((uint32_t)LZMAT_GET_U4(pbIn,inPos,cur_nib)<<14))
							       +0x4441;
							break;
					}
				}
				else
				{
					dist = cflag>>1;
					if(cflag&1)
					{
						inPos+=cur_nib;
						dist = (dist&0x7FF)+0x81;
						cur_nib^=1;
					}
					else
						dist = (dist&0x7F)+1;
				}
#undef cflag
				r_cnt = LZMAT_GET_U4(pbIn,inPos,cur_nib);
				if(r_cnt!=0xF)
				{
					r_cnt += 3;
				}
				else
				{
					if((inPos+1+cur_nib)>cbIn)
						return LZMAT_STATUS_INTEGRITY_FAILURE+2;
					r_cnt = LZMAT_GET_U8(pbIn+inPos,cur_nib);
					inPos++;
					if(r_cnt!=0xFF)
					{
						r_cnt += LZMAT_DEFAULT_CNT;
					}
					else
					{
						if((inPos+2+cur_nib)>cbIn)
							return LZMAT_STATUS_INTEGRITY_FAILURE+3;
						r_cnt = LZMAT_GET_LE16(pbIn+inPos,cur_nib)+LZMAT_1BYTE_CNT;
						inPos+=2;
						if(r_cnt==LZMAT_2BYTE_CNT)
						{
							// copy chunk
							if(cur_nib)
							{
								r_cnt = ((uint32_t)pbIn[inPos-4]&0xFC)<<5;
								inPos++;
								cur_nib = 0;
							}
							else
							{
								r_cnt = (GET_LE16(pbIn+inPos-5)&0xFC0)<<1;
							}
							r_cnt+=(tag&0x7F)+4;
							r_cnt<<=1;
							if((outPos+(r_cnt<<2))>cbOutBuf)
								return LZMAT_STATUS_BUFFER_TOO_SMALL;
							while(r_cnt-- && outPos<cbOutBuf)
							{
								*(uint32_t *)(pbOut+outPos)=*(uint32_t *)(pbIn+inPos);
								inPos+=4;
								outPos+=4;
							}
							break;
						}
					}
				}
				if(outPos<dist)
					return LZMAT_STATUS_INTEGRITY_FAILURE+4;
				if((outPos+r_cnt)>cbOutBuf)
					return LZMAT_STATUS_BUFFER_TOO_SMALL+1;
				r_pos = outPos-dist;
				while(r_cnt-- && outPos<cbOutBuf)
					pbOut[outPos++]=pbOut[r_pos++];
			}
			else
			{
				pbOut[outPos++]=LZMAT_GET_U8(pbIn+inPos,cur_nib);
				inPos++;
			}
		}
	}
	*pcbOut = outPos;
	return LZMAT_STATUS_OK;
}

#endif

