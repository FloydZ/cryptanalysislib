#ifndef CRYPTANALYSISLIB_TERNAY_LIST_ENUMERATION_H
#define CRYPTANALYSISLIB_TERNAY_LIST_ENUMERATION_H

#import <cstdint>
#import "list/enumeration/enumeration.h"


template<class List_Type, typename ChangeList_Type>
class ListEnumeration_Ternary {
public:
	/// needed type definitions
	using List = List_Type;
	using ChangeList = ChangeList_Type;

	/// needed variables
	List L1, L2;
	ChangeList cL1, cL2;
	ListIteration listIteration;

	void PrepareLists() noexcept {
		// variables
		constexpr uint64_t len = cceil(double(k+l)/2.);
		constexpr uint64_t offset = k+l-len;

		switch (listIteration) {
			case SingleFullLength: {
				// reset the changelist.
				cL1.resize(L1.size());
				cL2.clear();    // reset not going to be used.

				// variables
				Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{k+l, config.nr1, 0};
				Value data; data.zero();
				ChangeElement cl;
				int64_t oot, ctr = 0;
				cl.first = 0; cl.second = 1;

				cctr.left_single_init(data, config.symbol);

				do {
					// save the current value.
					L1.data_value(ctr) = data;
					cL1[ctr] = cl;

					// fetch the new value
					oot = cctr.left_single_step(data, cl);
					ctr += 1;
				} while(oot != 0);

				break;
			}
			case MultiFullLength: {
				// reset the changelist.
				cL1.resize(L1.size());
				cL2.clear();    // reset not going to be used.

				// variables
				Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{k+l, config.nr1, config.nr2};
				Value data; data.zero();
				ChangeElement cl;
				int64_t oot = 0, ctr = 0;
				cl.first = 0; cl.second = 1;

				cctr.left_init(data);

				do {
					// save the current value.
					L1.data_value(ctr) = data;
					cL1[ctr] = cl;

					// fetch the new value
					oot = cctr.left_step(data, cl);
					ctr += 1;
				} while(oot != 0);

				break;
			}
			case SinglePartialSingle : {
				constexpr uint64_t LeftRightSplit = k+l-config.alpha;
				constexpr uint64_t LeftRightSplitLen = cceil(double(LeftRightSplit)/2.);
				constexpr uint64_t RepsLessLen = config.alpha;
				constexpr uint64_t RepsLessPartLen = cceil(RepsLessLen/4.);

				static_assert(a == 0 ? (config.nr2 == 0) : true);
				static_assert(a == k+l ? (config.nr1 == 0) : true);
				static_assert(LeftRightSplitLen >= config.nr1);
				static_assert(RepsLessPartLen >= config.nr2);


				cL1.resize(bc(LeftRightSplitLen, config.nr1));
				cL2.resize(bc(RepsLessPartLen, config.nr2));

				Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{LeftRightSplitLen, config.nr1, 0};
				Combinations_Chase_TernaryRow<Value, ChangeElement> CCTRRepsLessPart{RepsLessPartLen, config.nr2, 0};

				Value data, data2; data.zero(); data2.zero();
				Value RepsLessPartData0, RepsLessPartData1, RepsLessPartData2, RepsLessPartData3, RepsLessPartData4;
				RepsLessPartData0.zero(); RepsLessPartData1.zero(); RepsLessPartData2.zero(); RepsLessPartData3.zero(); RepsLessPartData4.zero();

				ChangeElement CL, RepsLessPartCL;
				int64_t oot, oot2, ctr1 = 0, ctr = 0;
				CL.first = 0; CL.second = 1;
				RepsLessPartCL.first = 0; RepsLessPartCL.second = 1;

				if constexpr(RepsLessPartLen != 0) {
					CCTRRepsLessPart.left_single_init(RepsLessPartData0, config.symbol);
				}
				if constexpr(LeftRightSplitLen != 0) {
					cctr.left_single_init(data, config.symbol);
				}

				bool already_set = false;
				do {
					// save the current value.
					data2 = data;
					data2.left_shift(LeftRightSplitLen);

					do {
						RepsLessPartData1 = RepsLessPartData0;
						RepsLessPartData1.left_shift(LeftRightSplit);

						RepsLessPartData2 = RepsLessPartData1;
						RepsLessPartData2.left_shift(RepsLessPartLen);

						RepsLessPartData3 = RepsLessPartData2;
						RepsLessPartData3.left_shift(RepsLessPartLen);

						RepsLessPartData4 = RepsLessPartData0;
						RepsLessPartData4.left_shift(k+l-RepsLessPartLen);

						Value::add(L1.data_value(ctr), data,  RepsLessPartData1);
						Value::add(L2.data_value(ctr), data2, RepsLessPartData2);
						Value::add(L3.data_value(ctr), data,  RepsLessPartData3);
						Value::add(L4.data_value(ctr), data2, RepsLessPartData4);

						if (!already_set)
							cL2[ctr] = RepsLessPartCL;

						//std::cout << L1.data_value(ctr) << " | " << L2.data_value(ctr) << " | " << L3.data_value(ctr) << " | " << L4.data_value(ctr) << ", " << CL.first << ":" << CL.second << ", " << RepsLessPartCL.first << ":" << RepsLessPartCL.second << "\n";
						oot2 = CCTRRepsLessPart.left_single_step(RepsLessPartData0, RepsLessPartCL);
						ctr += 1;
					} while (oot2 != 0);

					already_set = true;
					RepsLessPartData0.zero();
					CCTRRepsLessPart.left_single_init(RepsLessPartData0, config.symbol);

					if constexpr(LeftRightSplitLen != 0) {
						// fetch the new value
						cL1[ctr1] = CL;
						//std::cout << CL.first << ":" << CL.second << "\n";

						oot = cctr.left_single_step(data, CL);
						ctr1 += 1;
					} else {
						oot = 0;
					}
				} while(oot != 0);

				break;
			}
			case MultiDisjointBlock : { std::cout << "not impl\n"; break;}
			case MITMSingle : {
				// reset the changelist. Only one changelist i dont care
				// create the second baselist with + ceil(double(k+l)/2.)
				cL1.resize(L1.size());
				cL2.clear();    // reset not going to be used.

				Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{len, config.nr1, 0};
				Value data, data2; data.zero();
				ChangeElement cl;
				int64_t oot, ctr = 0;
				cl.first = 0; cl.second = 1;

				cctr.left_single_init(data, config.symbol);

				do {
					// save the current value.
					data2 = data;
					data2.left_shift(offset);
					L1.data_value(ctr) = data;
					L2.data_value(ctr) = data2;
					cL1[ctr] = cl;

					//std::cout << data << " | " << data2 << ", " << cl.first << ":" << cl.second << "\n";

					// fetch the new value
					oot = cctr.left_single_step(data, cl);
					ctr += 1;
				} while(oot != 0);

				break;
			}
			case MITMMulti : {
				// reset the changelist.
				cL1.resize(L1.size());
				cL2.clear();    // reset not going to be used.
				Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{len, config.nr1, config.nr2};
				Value data, data2; data.zero();
				ChangeElement cl;
				int64_t oot, ctr = 0;
				cl.first = 0; cl.second = 1;

				cctr.left_init(data);

				do {
					// save the current value.
					data2 = data;
					data2.left_shift(offset);
					L1.data_value(ctr) = data;
					L2.data_value(ctr) = data2;
					cL1[ctr] = cl;

					// fetch the new value
					oot = cctr.left_step(data, cl);
					ctr += 1;
				} while(oot != 0);

				break;
			}
			case MITMEnumSingle: {
				// reset the changelist. Only one changelist i dont care
				// create the second baselist with + ceil(double(k+l)/2.)
				cL1.resize(L1.size());
				cL2.clear();    // reset not going to be used.
				uint64_t ctr = 0;

				//iterate over every weight p
				for (uint32_t cp = 1; cp <= config.nr1; ++cp) {
					Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{len, cp, 0};
					Value data, data2; data.zero();
					ChangeElement cl;
					int64_t oot;
					cl.first = 0; cl.second = 1;

					cctr.left_single_init(data, config.symbol);

					do {
						// save the current value.
						data2 = data;
						data2.left_shift(offset);
						L1.data_value(ctr) = data;
						L2.data_value(ctr) = data2;
						cL1[ctr] = cl;

						//std::cout << data << " | " << data2 << ", " << cl.first << ":" << cl.second << "\n";

						// fetch the new value
						oot = cctr.left_single_step(data, cl);
						ctr += 1;
					} while(oot != 0);
				}


				break;
			}
			case EnumSinglePartialSingle: {
				constexpr uint64_t LeftRightSplit = k+l-a;
				constexpr uint64_t LeftRightSplitLen = cceil(double(LeftRightSplit)/2.);
				constexpr uint64_t RepsLessLen = config.alpha;
				constexpr uint64_t RepsLessPartLen = cceil(double(RepsLessLen)/4.);

				//constexpr uint64_t sbc1 = sum_bc(LeftRightSplitLen, config.nr1);
				constexpr uint64_t sbc2 = sum_bc(RepsLessPartLen, config.nr2);

				static_assert(a == 0 ? (config.nr2 == 0) : true);
				static_assert(a == k+l ? (config.nr1 == 0) : true);
				static_assert(LeftRightSplitLen >= config.nr1);
				static_assert(RepsLessPartLen >= config.nr2);

				/// copy nr_blocks*(len)*sizeof(Value) Bytes from in to out
				auto copyBlock = [](List L,
									const uint64_t in,
									const uint64_t out,
									const uint64_t len,
									const uint64_t nr_blocks) {
				  for (uint64_t i = 0; i < nr_blocks; ++i) {
					  memcpy(L.data_value() + out + i*len, L.data_value() + in, len * sizeof(Value));
				  }
				};

				// first iterate over the disjunctive part
				if(RepsLessPartLen != 0) {
					for (uint32_t cp = 1; cp <= config.nr2; ++cp) {
						const uint64_t nr_elements = bc(RepsLessPartLen, cp);
						const uint64_t off1 = LeftRightSplitLen == 0 ? 1 : bc(LeftRightSplitLen, 1);
						uint64_t ctr = cp == 1 ? 0 : off1 * sum_bc(RepsLessPartLen, cp - 1);
						uint64_t sctr = ctr;

						ccL2[cp - 1].resize(nr_elements);

						Value RepsLessPartData0, RepsLessPartData1, RepsLessPartData2, RepsLessPartData3, RepsLessPartData4;
						RepsLessPartData0.zero();
						RepsLessPartData1.zero();
						RepsLessPartData2.zero();
						RepsLessPartData3.zero();
						RepsLessPartData4.zero();
						Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{RepsLessPartLen, cp, 0};
						cctr.left_single_init(RepsLessPartData0, config.symbol);

						ChangeElement cl;
						cl.first = 0;
						cl.second = 1;

						int64_t oot;

						do {
							RepsLessPartData1 = RepsLessPartData0;
							RepsLessPartData1.left_shift(LeftRightSplit);

							RepsLessPartData2 = RepsLessPartData1;
							RepsLessPartData2.left_shift(RepsLessPartLen);

							RepsLessPartData3 = RepsLessPartData2;
							RepsLessPartData3.left_shift(RepsLessPartLen);

							RepsLessPartData4 = RepsLessPartData0;
							RepsLessPartData4.left_shift(k + l - RepsLessPartLen);

							// save the current value.
							L1.data_value(ctr) = RepsLessPartData1;
							L2.data_value(ctr) = RepsLessPartData2;
							L3.data_value(ctr) = RepsLessPartData3;
							L4.data_value(ctr) = RepsLessPartData4;

							// save the change list
							ccL2[cp - 1][ctr - sctr] = cl;

							// fetch the new value
							oot = cctr.left_single_step(RepsLessPartData0, cl);
							ctr += 1;
						} while (oot != 0);

						// copy the calculated disjunctive part to each iteration of the mitm part
						if (LeftRightSplitLen != 0) {
							for (uint32_t i = 1; i <= config.nr1; ++i) {
								const uint64_t nr_copies = bc(LeftRightSplitLen, i);
								const uint64_t out = sbc2 * (i == 1 ? 0 : sum_bc(LeftRightSplitLen, i - 1));
								const uint64_t out1 = cp == 1 ? 0 : nr_copies * sum_bc(RepsLessPartLen, cp - 1);
								const uint64_t out2 = out + out1;

								copyBlock(L1, sctr, out2, nr_elements, nr_copies);
								copyBlock(L2, sctr, out2, nr_elements, nr_copies);
								copyBlock(L3, sctr, out2, nr_elements, nr_copies);
								copyBlock(L4, sctr, out2, nr_elements, nr_copies);
							}
						}
					}
				}

				// iterate over every weight p in the MITM part
				if (LeftRightSplitLen != 0) {
					for (uint32_t cp = 1; cp <= config.nr1; ++cp) {
						const uint64_t nr_elements = bc(LeftRightSplitLen, cp);
						ccL1[cp - 1].resize(nr_elements);

						Value data, data2;
						data.zero();
						Combinations_Chase_TernaryRow<Value, ChangeElement> cctr{LeftRightSplitLen, cp, 0};
						cctr.left_single_init(data, config.symbol);

						ChangeElement cl;
						cl.first = 0;
						cl.second = 1;

						int64_t oot;
						uint64_t ctr2 = 0;

						do {
							// save the current value.
							data2 = data;
							data2.left_shift(LeftRightSplitLen);

							const uint64_t out = sbc2 * (cp == 1 ? 0 : sum_bc(LeftRightSplitLen, cp - 1));

							for (uint64_t cp2 = 1; cp2 <= config.nr2; ++cp2) {
								uint64_t out1 = nr_elements * (cp2 == 1 ? 0 : sum_bc(RepsLessPartLen, cp2 - 1));
								uint64_t out2 = ctr2 * bc(RepsLessPartLen, cp2);
								uint64_t out3 = out + out1 + out2;
								for (uint64_t j = 0; j < bc(RepsLessPartLen, cp2); ++j) {
									Value::add(L1.data_value(out3 + j), L1.data_value(out3 + j), data);
									Value::add(L2.data_value(out3 + j), L2.data_value(out3 + j), data2);
									Value::add(L3.data_value(out3 + j), L3.data_value(out3 + j), data);
									Value::add(L4.data_value(out3 + j), L4.data_value(out3 + j), data2);
								}
							}

							ccL1[cp - 1][ctr2] = cl;

							//std::cout << data << " | " << data2 << ", " << cl.first << ":" << cl.second << "\n";
							// fetch the new value
							oot = cctr.left_single_step(data, cl);
							ctr2 += 1;
						} while (oot != 0);
					}
				}


				//				uint64_t clctr1 = 0, i_clctr1 = 0, clctr2 = 0, i_clctr2 = 0;
				//				for (uint64_t i = 0; i < BaseList1Size; ++i) {
				//					std::cout << i << ":\t" << L1.data_value(i) << " | " << L2.data_value(i) << " | " << L3.data_value(i) << " | " << L4.data_value(i) << "\t";
				//					if (LeftRightSplitLen != 0) {
				//						std::cout << ccL1[clctr1][i_clctr1].first << ":" << ccL1[clctr1][i_clctr1].second;
				//					}
				//
				//					if (RepsLessPartLen != 0) {
				//						std::cout << " " << ccL2[clctr2][i_clctr2].first << ":" << ccL2[clctr2][i_clctr2].second;
				//					}
				//
				//					std::cout << ", " << clctr1 << ", " << i_clctr1 << ", " << clctr2 << ", " << i_clctr2 << "\n";
				//
				//					i_clctr2 += 1;
				//					if (i_clctr2 == ccL2[clctr2].size()) {
				//						i_clctr2 = 0;
				//						i_clctr1 += 1;
				//
				//						if (i_clctr1 == ccL1[clctr1].size()) {
				//							i_clctr1 = 0;
				//							i_clctr2 = 0;
				//							clctr2 += 1;
				//						}
				//					}
				//
				//					if (clctr2 == config.nr2) {
				//						clctr1 += 1;
				//						clctr2 = 0;
				//					}
				//				}
				break;
			}
		}
	}


	// only valid for "SinglePartialSingle"
	void CartesianProduct(const ChangeList &cL1, const ChangeList &cL2,
						  const uint64_t start, const uint32_t pprime1,
						  const uint32_t pprime2, const uint32_t tid) noexcept {
		uint64_t spos2 = start;          //start_pos(tid);
		uint64_t epos2 = start+cL2.size(); // end_pos(tid);

		constexpr uint64_t LeftRightSplit       = k+l-config.alpha;
		constexpr uint64_t LeftRightSplitLen    = cceil(double(LeftRightSplit)/2.);
		constexpr uint64_t RepsLessLen          = k+l-LeftRightSplit;
		constexpr uint64_t RepsLessPartLen      = cceil(RepsLessLen/4.);

		auto copyBlock = [start](List L, uint64_t in, uint64_t len){
		  memcpy(L.data_label() + start + ((in+1)*len), L.data_label() + start + (in*len), len*sizeof(Label));
		  //const uint64_t val = start + in*len;
		  //std::copy(L.begin() + val + len,
		  //		  L.begin() + start,
		  //		  L.begin() + val + len);
		};

		// Set the first element.
		L1.data_label(spos2).zero();
		L2.data_label(spos2).zero();
		L3.data_label(spos2).zero();
		L4.data_label(spos2).zero();

		// write the first element in the disjunctive part
		if (RepsLessPartLen != 0) {
			for (uint32_t i = 0; i < pprime2; ++i) {
				// set the RepsLess part.
				Label::add(L1.data_label(spos2), L1.data_label(spos2),
						   HT.__data[i + LeftRightSplit + RepsLessPartLen * 0]);
				Label::add(L2.data_label(spos2), L2.data_label(spos2),
						   HT.__data[i + LeftRightSplit + RepsLessPartLen * 1]);
				Label::add(L3.data_label(spos2), L3.data_label(spos2),
						   HT.__data[i + LeftRightSplit + RepsLessPartLen * 2]);
				Label::add(L4.data_label(spos2), L4.data_label(spos2), HT.__data[i + k + l - RepsLessPartLen]);
			}
		}

		// create the first label of the MITM part
		Label tmp1, tmp2, tmp3, tmp4;
		if (LeftRightSplitLen != 0) {
			for (uint32_t i = 0; i < pprime1; ++i) {
				Label::add(tmp1, tmp1, HT.__data[i]);
				Label::add(tmp2, tmp2, HT.__data[i + LeftRightSplitLen]);
				Label::add(tmp3, tmp3, HT.__data[i]);
				Label::add(tmp4, tmp4, HT.__data[i + LeftRightSplitLen]);
			}
		}

		// calculate the all labels in the disjunct part.
		if (RepsLessPartLen != 0) {
			for (uint64_t i = spos2 + 1; i < epos2; ++i) {
				L1.data_label()[i] = L1.data_label()[i - 1];
				L2.data_label()[i] = L2.data_label()[i - 1];
				L3.data_label()[i] = L3.data_label()[i - 1];
				L4.data_label()[i] = L4.data_label()[i - 1];

				Label::sub(L1.data_label(i), L1.data_label(i),
						   HT.__data[cL2[i - spos2].first + LeftRightSplit + RepsLessPartLen * 0]);
				Label::sub(L2.data_label(i), L2.data_label(i),
						   HT.__data[cL2[i - spos2].first + LeftRightSplit + RepsLessPartLen * 1]);
				Label::sub(L3.data_label(i), L3.data_label(i),
						   HT.__data[cL2[i - spos2].first + LeftRightSplit + RepsLessPartLen * 2]);
				Label::sub(L4.data_label(i), L4.data_label(i),
						   HT.__data[cL2[i - spos2].first + k + l - RepsLessPartLen]);

				Label::add(L1.data_label(i), L1.data_label(i),
						   HT.__data[cL2[i - spos2].second + LeftRightSplit + RepsLessPartLen * 0]);
				Label::add(L2.data_label(i), L2.data_label(i),
						   HT.__data[cL2[i - spos2].second + LeftRightSplit + RepsLessPartLen * 1]);
				Label::add(L3.data_label(i), L3.data_label(i),
						   HT.__data[cL2[i - spos2].second + LeftRightSplit + RepsLessPartLen * 2]);
				Label::add(L4.data_label(i), L4.data_label(i),
						   HT.__data[cL2[i - spos2].second + k + l - RepsLessPartLen]);
			}
		}

		uint64_t spos1 = start;
		uint64_t epos1 = start+bc(RepsLessPartLen, pprime2);

		// now calc the MITM part
		if  (LeftRightSplitLen != 0) {
			for (uint64_t i = 0; i < cL1.size() - 1; ++i) {
				copyBlock(L1, i, cL2.size());
				for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L1.data_label(j), L1.data_label(j), tmp1); }
				Label::sub(tmp1, tmp1, HT.__data[cL1[i + 1].first]);
				Label::add(tmp1, tmp1, HT.__data[cL1[i + 1].second]);

				copyBlock(L2, i, cL2.size());
				for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L2.data_label(j), L2.data_label(j), tmp2); }
				Label::sub(tmp2, tmp2, HT.__data[cL1[i + 1].first + LeftRightSplitLen]);
				Label::add(tmp2, tmp2, HT.__data[cL1[i + 1].second + LeftRightSplitLen]);

				copyBlock(L3, i, cL2.size());
				for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L3.data_label(j), L3.data_label(j), tmp3); }
				Label::sub(tmp3, tmp3, HT.__data[cL1[i + 1].first]);
				Label::add(tmp3, tmp3, HT.__data[cL1[i + 1].second]);

				copyBlock(L4, i, cL2.size());
				for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L4.data_label(j), L4.data_label(j), tmp4); }
				Label::sub(tmp4, tmp4, HT.__data[cL1[i + 1].first + LeftRightSplitLen]);
				Label::add(tmp4, tmp4, HT.__data[cL1[i + 1].second + LeftRightSplitLen]);

				spos1 += bc(RepsLessPartLen, pprime2);
				epos1 += bc(RepsLessPartLen, pprime2);
			}
		}

		// final add
		for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L1.data_label(j), L1.data_label(j), tmp1); }
		for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L2.data_label(j), L2.data_label(j), tmp2); }
		for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L3.data_label(j), L3.data_label(j), tmp3); }
		for (uint64_t j = spos1; j < epos1; ++j) { Label::add(L4.data_label(j), L4.data_label(j), tmp4); }
	}

	// given a position within the list, this function returns a positions corresponding to this changelist created
	// this element.
	void TranslatePos2ChangeList(size_t &pos1, size_t &pos2,
								 size_t &end1, size_t &end2,
								 const size_t pos) noexcept {
		switch (config.listIteration) {
			case SingleFullLength: {
				pos1 = pos;
				pos2 = 0;
				break;
			}
			case MultiFullLength: {
				std::cout << "not implemented\n";
				break;
			}
			case SinglePartialSingle: {
				pos1 = pos/cL1.size();
				pos2 = (pos%cL1.size())/cL2.size();
				break;
			}
			case MultiDisjointBlock: {
				std::cout << "not implemented\n";
				break;
			}
			case MITMSingle: {
				std::cout << "not implemented\n";
				break;
			}
			case MITMMulti: {
				std::cout << "not implemented\n";
				break;
			}
			case MITMEnumSingle: {
				std::cout << "not implemented\n";
				break;
			}
			case EnumSinglePartialSingle: {
				std::cout << "not implemented\n";
				break;
			}
		}
	}

	void FillLists(const uint32_t tid) noexcept {
		// first/last element to work on for each thread
		const size_t spos = start_pos(tid);
		const size_t epos = end_pos(tid);

		// helper values
		constexpr uint64_t len = cceil(double(k+l)/2.);
		constexpr uint64_t offset = k+l-len;

		// length of the mitm part. => Reps part
		constexpr uint64_t LeftRightSplit = k+l-config.alpha;
		constexpr uint64_t LeftRightSplitLen = cceil(double(LeftRightSplit)/2.);

		// length of the disjunct part
		constexpr uint64_t RepsLessLen = k+l-LeftRightSplit;
		constexpr uint64_t RepsLessPartLen = cceil(double(RepsLessLen)/4.);

		constexpr uint32_t pp = config.nr1 + config.nr2;
		// array holding the position of the bits set of the first element each thread starts to work on.
		uint16_t P1[pp] = {0}, P2[pp] = {0}, P3[pp] = {0}, P4[pp] = {0};

		// extract the bits currently set in the value
		L1.data_value()[spos].get_bits_set(P1, pp); L2.data_value()[spos].get_bits_set(P2, pp);
		L3.data_value()[spos].get_bits_set(P3, pp); L4.data_value()[spos].get_bits_set(P4, pp);

		// TODO multithreading
		switch (config.listIteration) {
			case SingleFullLength: {
				// Set the first element.
				L1.data_label(spos).zero();
				for (uint32_t i = 0; i < config.nr1; ++i) {
					for (uint32_t j = 0; j < config.symbol; ++j) {
						Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i]]);
					}
				}

				// set the remaining elements.
				for (size_t i = spos + 1; i < epos; ++i) {
					L1.data_label()[i] = L1.data_label()[i-1];

					for (uint32_t j = 0; j < config.symbol; ++j) {
						Label::add(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].first]);
						Label::add(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].second]);
					}
				}

				// we are lazy, hopefully that`s copy
				L2 = L1;
				OMP_BARRIER

				// Jeah stupid, but I couldn't think of something smarter.
				for(size_t j = spos; j < epos; j++) {
					L2.data_label(j).template neg<n-k-l, n-k-l+l1>();
				}

				break;
			}
			case MultiFullLength: {
				std::cout << "not implemented\n";
				break;
			}
			case SinglePartialSingle: {
				//      LeftRightSplitLen   LeftRightSplit
				//  [            |                |       |       |       |       ]
				//                            k+l-alpha
				// clear the first element.
				L1.data_label(spos).zero(); L2.data_label(spos).zero();
				L3.data_label(spos).zero(); L4.data_label(spos).zero();

				// set the first element.
				for (uint32_t i = 0; i < config.nr1; ++i) {
					for (uint32_t j = 0; j < config.symbol; ++j) {
						// set the Left/Right split part.
						//	Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i]]);
						//	Label::add(L2.data_label(spos), L2.data_label(spos), HT.__data[P1[i] + LeftRightSplitLen]);
						//	Label::add(L3.data_label(spos), L3.data_label(spos), HT.__data[P1[i]]);
						//	Label::add(L4.data_label(spos), L4.data_label(spos), HT.__data[P1[i] + LeftRightSplitLen]);

						Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i]]);
						Label::add(L2.data_label(spos), L2.data_label(spos), HT.__data[P2[i]]);
						Label::add(L3.data_label(spos), L3.data_label(spos), HT.__data[P3[i]]);
						Label::add(L4.data_label(spos), L4.data_label(spos), HT.__data[P4[i]]);
					}
				}

				// still setting the first element, but now the right part = reps less
				for (uint32_t i = config.nr1; i < config.nr1+config.nr2; ++i) {
					for (uint32_t j = 0; j < config.symbol; ++j) {
						// set the RepsLess part.
						//	Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i] + LeftRightSplit + RepsLessPartLen * 0]);
						//	Label::add(L2.data_label(spos), L2.data_label(spos), HT.__data[P1[i] + LeftRightSplit + RepsLessPartLen * 1]);
						//	Label::add(L3.data_label(spos), L3.data_label(spos), HT.__data[P1[i] + LeftRightSplit + RepsLessPartLen * 2]);
						//	Label::add(L4.data_label(spos), L4.data_label(spos), HT.__data[P1[i] + k + l - RepsLessPartLen]);

						Label::add(L1.data_label(spos), L1.data_label(spos), HT.__data[P1[i]]);
						Label::add(L2.data_label(spos), L2.data_label(spos), HT.__data[P2[i]]);
						Label::add(L3.data_label(spos), L3.data_label(spos), HT.__data[P3[i]]);
						Label::add(L4.data_label(spos), L4.data_label(spos), HT.__data[P4[i]]);
					}
				}

				// set the limits of the work for the current thread.
				size_t LeftRight_begin = spos + 1,
						RepsLess_begin = 1,
						LeftRight_end = epos /*cL1.size()*/,
						RepsLess_end = cL2.size();

				// TODO
				//if constexpr(config.threads != 1) {
				//	TranslatePos2ChangeList(LeftRight_begin, RepsLess_begin, LeftRight_end, RepsLess_end, spos+1);
				//}

				for (size_t LeftRight_ctr = LeftRight_begin; LeftRight_ctr <= LeftRight_end; ++LeftRight_ctr) {
					const size_t i = LeftRight_ctr*cL2.size();

					// now set the elements for the RepsPart.
					for (size_t RepsLess_ctr = RepsLess_begin; RepsLess_ctr < RepsLess_end; ++RepsLess_ctr) {
						const size_t j = i - cL2.size() + RepsLess_ctr;
						ASSERT((cL2[RepsLess_ctr].first  + LeftRightSplit + RepsLessPartLen * 2) < (k+l));
						ASSERT((cL2[RepsLess_ctr].second + LeftRightSplit + RepsLessPartLen * 2) < (k+l));

						// first copy everything.
						L1.data_label()[j] = L1.data_label()[j-1];
						L2.data_label()[j] = L2.data_label()[j-1];
						L3.data_label()[j] = L3.data_label()[j-1];
						L4.data_label()[j] = L4.data_label()[j-1];

						// subtract the first one
						Label::sub(L1.data_label(j), L1.data_label(j), HT.__data[cL2[RepsLess_ctr].first + LeftRightSplit + RepsLessPartLen * 0]);
						Label::sub(L2.data_label(j), L2.data_label(j), HT.__data[cL2[RepsLess_ctr].first + LeftRightSplit + RepsLessPartLen * 1]);
						Label::sub(L3.data_label(j), L3.data_label(j), HT.__data[cL2[RepsLess_ctr].first + LeftRightSplit + RepsLessPartLen * 2]);
						Label::sub(L4.data_label(j), L4.data_label(j), HT.__data[cL2[RepsLess_ctr].first + k + l - RepsLessPartLen]);

						// and add the next one.
						Label::add(L1.data_label(j), L1.data_label(j), HT.__data[cL2[RepsLess_ctr].second + LeftRightSplit + RepsLessPartLen * 0]);
						Label::add(L2.data_label(j), L2.data_label(j), HT.__data[cL2[RepsLess_ctr].second + LeftRightSplit + RepsLessPartLen * 1]);
						Label::add(L3.data_label(j), L3.data_label(j), HT.__data[cL2[RepsLess_ctr].second + LeftRightSplit + RepsLessPartLen * 2]);
						Label::add(L4.data_label(j), L4.data_label(j), HT.__data[cL2[RepsLess_ctr].second + k + l - RepsLessPartLen]);
					}

					// break if we finished the last iterations
					if (LeftRight_ctr == cL1.size())
						break;

					// reset the beginning
					RepsLess_begin = 1;


					// now make a step in the LeftRightPart
					L1.data_label()[i] = L1.data_label()[i-1];
					L2.data_label()[i] = L2.data_label()[i-1];
					L3.data_label()[i] = L3.data_label()[i-1];
					L4.data_label()[i] = L4.data_label()[i-1];

					// but first clear the upper coordinates. E.g. subtract the last element of the inner loop.
					for (uint32_t j = 1; j < config.nr2+1; ++j) {
						Label::sub(L1.data_label()[i], L1.data_label()[i], HT.__data[LeftRightSplit + RepsLessPartLen * 1 - j]);
						Label::sub(L2.data_label()[i], L2.data_label()[i], HT.__data[LeftRightSplit + RepsLessPartLen * 2 - j]);
						Label::sub(L3.data_label()[i], L3.data_label()[i], HT.__data[LeftRightSplit + RepsLessPartLen * 3 - j]);
						Label::sub(L4.data_label()[i], L4.data_label()[i], HT.__data[k + l - j]);
					}

					// and final add the next Left/Right part on it.
					for (uint32_t j = 0; j < config.symbol; ++j) {
						ASSERT((cL1[LeftRight_ctr].first +LeftRightSplitLen) < (k+l));
						ASSERT((cL1[LeftRight_ctr].second+LeftRightSplitLen) < (k+l));

						Label::sub(L1.data_label(i), L1.data_label(i), HT.__data[cL1[LeftRight_ctr].first]);
						Label::add(L1.data_label(i), L1.data_label(i), HT.__data[cL1[LeftRight_ctr].second]);
						Label::sub(L2.data_label(i), L2.data_label(i), HT.__data[cL1[LeftRight_ctr].first  + LeftRightSplitLen]);
						Label::add(L2.data_label(i), L2.data_label(i), HT.__data[cL1[LeftRight_ctr].second + LeftRightSplitLen]);
						Label::sub(L3.data_label(i), L3.data_label(i), HT.__data[cL1[LeftRight_ctr].first]);
						Label::add(L3.data_label(i), L3.data_label(i), HT.__data[cL1[LeftRight_ctr].second]);
						Label::sub(L4.data_label(i), L4.data_label(i), HT.__data[cL1[LeftRight_ctr].first  + LeftRightSplitLen]);
						Label::add(L4.data_label(i), L4.data_label(i), HT.__data[cL1[LeftRight_ctr].second + LeftRightSplitLen]);
					}

					for (uint32_t j = 0; j < config.nr2; ++j) {
						// ok nicht perfekt. vll kann man das weg optimieren
						Label::add(L1.data_label(i), L1.data_label(i), HT.__data[LeftRightSplit + RepsLessPartLen * 0 + j]);
						Label::add(L2.data_label(i), L2.data_label(i), HT.__data[LeftRightSplit + RepsLessPartLen * 1 + j]);
						Label::add(L3.data_label(i), L3.data_label(i), HT.__data[LeftRightSplit + RepsLessPartLen * 2 + j]);
						Label::add(L4.data_label(i), L4.data_label(i), HT.__data[k + l - RepsLessPartLen + j]);
					}
				}

				OMP_BARRIER
				// TODO optimize
				for(size_t j = spos; j < epos; j++) {
					L3.data_label(j).template neg<n-k-l+l1, n-k>();
					L2.data_label(j).template neg<n-k-l, n-k-l+l1>();
					L4.data_label(j).template neg<n-k-l, n-k>();
				}

				break;
			}
			case MultiDisjointBlock: {
				std::cout << "not implemented\n";
				break;
			}
			case MITMSingle: {
				// Set the first element.
				L1.data_label(spos).zero();
				L2.data_label(spos).zero();

				for (uint32_t i = 0; i < config.nr1; ++i) {
					for (uint32_t j = 0; j < config.symbol; ++j) {
						Label::add(L1.data_label(0), L1.data_label(0), HT.__data[P1[i]]);
						Label::add(L2.data_label(0), L2.data_label(0), HT.__data[P1[i] + offset]);
					}
				}

				// set the remaining elements.
				for (size_t i = spos + 1; i < epos; ++i) {
					L1.data_label()[i] = L1.data_label()[i-1];
					L2.data_label()[i] = L2.data_label()[i-1];

					for (uint32_t j = 0; j < config.symbol; ++j) {
						ASSERT((cL1[i].first +offset) < (k+l));
						ASSERT((cL1[i].second+offset) < (k+l));

						Label::sub(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].first]);
						Label::add(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].second]);

						Label::sub(L2.data_label(i), L2.data_label(i), HT.__data[cL1[i].first + offset]);
						Label::add(L2.data_label(i), L2.data_label(i), HT.__data[cL1[i].second + offset]);
					}
				}

				// TODO optimize
				for(size_t j = spos; j < epos; j++) {
					L2.data_label(j).neg(n-k-l, n-k);
				}

				break;
			}
			case MITMMulti: {
				// Some security measurements
				ASSERT(cL1.size() == L1.size());
				ASSERT(L1.size() == L2.size());

				// first some hard offsets.
				constexpr uint64_t NrOfOnes = bc(len, config.nr1);
				constexpr uint64_t NrOfTwos = bc(len-config.nr1, config.nr2);
				ASSERT(L1.size() == (NrOfOnes*NrOfTwos));

				// TODO make available for every thread. So precompute this thing for every thread, to safe mem.
				// build up lookup table for the sums of twos.
				std::array<Label, k+l> lookup;
				for (uint32_t i = 0; i < k+l; ++i) {
					Label::add(lookup[i], HT.__data[i], HT.__data[i]);
				}

				// Set the first element, assumed to be: [1...12...20...0]
				L1.data_label(spos).zero();
				L2.data_label(spos).zero();

				// first the ones...
				for (uint32_t i = 0; i < config.nr1; ++i) {
					Label::add(L1.data_label(0), L1.data_label(0), HT.__data[P1[i]]);
					Label::add(L2.data_label(0), L2.data_label(0), HT.__data[P1[i] + offset]);
				}

				// ... then the twos
				// TODO das ist noch nicht voll korrekt, weil die 2en nicht direkt hinter den einsen
				for (uint32_t i = config.nr1; i < + config.nr1+config.nr2; ++i) {
					Label::add(L1.data_label(0), L1.data_label(0), lookup[P1[i]]);
					Label::add(L2.data_label(0), L2.data_label(0), lookup[P1[i]+ offset]);
				}

				// set the remaining elements.
				for (uint32_t i = 1; i < (NrOfOnes*NrOfTwos); ++i) {
					while((i%NrOfTwos) != 0) {
						// copy the previous label
						L1.data_label()[i] = L1.data_label()[i-1];
						L2.data_label()[i] = L2.data_label()[i-1];

						// add the changing two
						Label::sub(L1.data_label(i), L1.data_label(i), lookup[cL1[i].first]);
						Label::add(L1.data_label(i), L1.data_label(i), lookup[cL1[i].second]);

						Label::sub(L2.data_label(i), L2.data_label(i), lookup[cL1[i].first  + offset]);
						Label::add(L2.data_label(i), L2.data_label(i), lookup[cL1[i].second + offset]);

						i += 1;
					}
					if (i >= (NrOfOnes*NrOfTwos))
						break;

					// Some security measurements
					ASSERT(i < cL1.size());
					ASSERT((cL1[i].first+offset) < (k+l));
					ASSERT((cL1[i].second+offset)< (k+l));

					L1.data_label()[i] = L1.data_label()[i-1];
					L2.data_label()[i] = L2.data_label()[i-1];

					// finished adding all the combinations generated by the twos. Now proceed with a change of the ones.
					// first clear the last twos
					for (uint32_t j = 0; j < config.nr2; ++j) {
						ASSERT(i-j > 0);
						ASSERT((i-j-1) < cL1.size());

						Label::sub(L1.data_label(i), L1.data_label(i), lookup[cL1[i-j-1].second]);
						Label::sub(L2.data_label(i), L2.data_label(i), lookup[cL1[i-j-1].second + offset]);
					}

					// then increase the ones
					Label::sub(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].first]);
					Label::add(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].second]);

					Label::sub(L2.data_label(i), L2.data_label(i), HT.__data[cL1[i].first  + offset]);
					Label::add(L2.data_label(i), L2.data_label(i), HT.__data[cL1[i].second + offset]);

					// and then add the first twos again.
					uint32_t twoctr = 0;
					for (uint32_t j = 0; (j < k+l) && (twoctr < config.nr2); ++j) {
						if (uint32_t(L1.data_value(i).get(j)) == 2u) {
							twoctr += 1;

							Label::add(L1.data_label(i), L1.data_label(i), lookup[j]);
							Label::add(L2.data_label(i), L2.data_label(i), lookup[j + offset]);
						}
					}
				}

				// TODO not correct.
				L3 = L1;
				L4 = L2;
				for(uint64_t j = spos; j < epos; j++) {
					L3.data_label(j).template neg<n-k-l+l1, n-k>();
					L2.data_label(j).template neg<n-k-l, n-k-l+l1>();
					L4.data_label(j).template neg<n-k-l, n-k>();
				}

				break;
			}
			case MITMEnumSingle: {
				// Set the first element.
				L1.data_label(spos).zero();
				L2.data_label(spos).zero();

				//iterate over every weight p
				for (uint32_t cp = 1; cp <= config.nr1; ++cp) {
					for (uint32_t i = 0; i < cp; ++i) {
						for (uint32_t j = 0; j < config.symbol; ++j) {
							Label::add(L1.data_label(0), L1.data_label(0), HT.__data[P1[i]]);
							Label::add(L2.data_label(0), L2.data_label(0), HT.__data[P1[i] + offset]);
						}
					}

					// set the remaining elements.
					for (size_t i = spos + 1; i < epos; ++i) {
						L1.data_label()[i] = L1.data_label()[i - 1];
						L2.data_label()[i] = L2.data_label()[i - 1];

						for (uint64_t j = 0; j < config.symbol; ++j) {
							ASSERT((cL1[i].first + offset) < (k + l));
							ASSERT((cL1[i].second + offset) < (k + l));

							Label::sub(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].first]);
							Label::add(L1.data_label(i), L1.data_label(i), HT.__data[cL1[i].second]);

							Label::sub(L2.data_label(i), L2.data_label(i), HT.__data[cL1[i].first + offset]);
							Label::add(L2.data_label(i), L2.data_label(i), HT.__data[cL1[i].second + offset]);
						}

						// std::cout << L1.data_label(i-1) << "->" << L1.data_label(i) << ", " << cL1[i].first << ":" << cL1[i].second << ", " << i << "\n";
					}
				}

				// TODO optimize
				for (size_t j = spos; j < epos; j++) {
					L2.data_label(j).neg(n - k - l, n - k);
				}

				break;
			}
			case EnumSinglePartialSingle: {
				size_t start = 0;
				if (LeftRightSplitLen != 0) {
					for (uint32_t cp1 = 1; cp1 <= config.nr1; ++cp1) {
						const size_t bc1 = bc(LeftRightSplitLen, cp1);
						for (uint32_t cp2 = 1; cp2 <= config.nr2; ++cp2) {
							CartesianProduct(ccL1[cp1 - 1], ccL2[cp2 - 1], start, cp1, cp2, tid);
							start += bc1 * bc(RepsLessPartLen, cp2);
						}
					}
				} else {
					for (uint32_t cp2 = 1; cp2 <= config.nr2; ++cp2) {
						CartesianProduct(ccL1[0], ccL2[cp2 - 1], start, 0, cp2, tid);
						start += bc(RepsLessPartLen, cp2);
					}
				}


				// FIXME optimize. currently this is for the n=100 instance responsible for 2%..
				for(size_t j = spos; j < epos; j++) {
					L3.data_label(j).template neg<n-k-l+l1, n-k>();
					L2.data_label(j).template neg<n-k-l, n-k-l+l1>();
					L4.data_label(j).template neg<n-k-l, n-k>();
				}

				break;
			}
		}
	}
};
#endif//CRYPTANALYSISLIB_TERNAY_H
