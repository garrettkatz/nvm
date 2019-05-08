register_names = ["r0","r1","r2","r3","r4","r5","rc"]

programs={
"myfirstprogram":""" 
    drf r3
	rem r0
	nxt
	ref r3	
	drf r4
	rem r1
	nxt
	ref r4
	sub add
	drf r5
	nxt
	mem r2
	ref r5
	sub start
    
start: drf r3
	rem r0
	cmp r0 /
	jie checknext
	nxt
	ref r3	
	drf r4
	rem r1
	cmp r1 /
	jie checkprev
	nxt
	ref r4
	sub add
	drf r5
	nxt
	mem r2
	ref r5
	sub start
	exit
	
checkprev: mov r1 0
           sub add
           drf r5
	       nxt
	       mem r2
	       ref r5
	       drf r3
	       rem r0
	       cmp r0 /
	       jie copyrc
	       nxt 
	       ref r3
	       sub checkprev  

checknext: drf r4
	       rem r1 
           cmp r1 /
           jie copyrc
           mov r0 0
           nxt
	       ref r4
	       sub add
           drf r5
	       nxt
	       mem r2
	       ref r5
	       sub checknext           
   
copyrc: drf r5
	nxt
	mem rc
	exit

add: cmp r0 1
	jie next_1

	cmp r0 2
	jie next_2

	cmp r0 3
	jie next_3

	cmp r0 4
	jie next_4

	cmp r0 5
	jie next_5

	cmp r0 6
	jie next_6

	cmp r0 7
	jie next_7

	cmp r0 8
	jie next_8

	cmp r0 9
	jie next_9

	cmp r0 0
	jie next_0



next_1: 	cmp r1 1
	jie sum_11

	cmp r1 2
	jie sum_12

	cmp r1 3
	jie sum_13

	cmp r1 4
	jie sum_14

	cmp r1 5
	jie sum_15

	cmp r1 6
	jie sum_16

	cmp r1 7
	jie sum_17

	cmp r1 8
	jie sum_18

	cmp r1 9
	jie sum_19

	cmp r1 0
	jie sum_10



next_2: 	cmp r1 1
	jie sum_21

	cmp r1 2
	jie sum_22

	cmp r1 3
	jie sum_23

	cmp r1 4
	jie sum_24

	cmp r1 5
	jie sum_25

	cmp r1 6
	jie sum_26

	cmp r1 7
	jie sum_27

	cmp r1 8
	jie sum_28

	cmp r1 9
	jie sum_29

	cmp r1 0
	jie sum_20



next_3: 	cmp r1 1
	jie sum_31

	cmp r1 2
	jie sum_32

	cmp r1 3
	jie sum_33

	cmp r1 4
	jie sum_34

	cmp r1 5
	jie sum_35

	cmp r1 6
	jie sum_36

	cmp r1 7
	jie sum_37

	cmp r1 8
	jie sum_38

	cmp r1 9
	jie sum_39

	cmp r1 0
	jie sum_30



next_4: 	cmp r1 1
	jie sum_41

	cmp r1 2
	jie sum_42

	cmp r1 3
	jie sum_43

	cmp r1 4
	jie sum_44

	cmp r1 5
	jie sum_45

	cmp r1 6
	jie sum_46

	cmp r1 7
	jie sum_47

	cmp r1 8
	jie sum_48

	cmp r1 9
	jie sum_49

	cmp r1 0
	jie sum_40



next_5: 	cmp r1 1
	jie sum_51

	cmp r1 2
	jie sum_52

	cmp r1 3
	jie sum_53

	cmp r1 4
	jie sum_54

	cmp r1 5
	jie sum_55

	cmp r1 6
	jie sum_56

	cmp r1 7
	jie sum_57

	cmp r1 8
	jie sum_58

	cmp r1 9
	jie sum_59

	cmp r1 0
	jie sum_50



next_6: 	cmp r1 1
	jie sum_61

	cmp r1 2
	jie sum_62

	cmp r1 3
	jie sum_63

	cmp r1 4
	jie sum_64

	cmp r1 5
	jie sum_65

	cmp r1 6
	jie sum_66

	cmp r1 7
	jie sum_67

	cmp r1 8
	jie sum_68

	cmp r1 9
	jie sum_69

	cmp r1 0
	jie sum_60



next_7: 	cmp r1 1
	jie sum_71

	cmp r1 2
	jie sum_72

	cmp r1 3
	jie sum_73

	cmp r1 4
	jie sum_74

	cmp r1 5
	jie sum_75

	cmp r1 6
	jie sum_76

	cmp r1 7
	jie sum_77

	cmp r1 8
	jie sum_78

	cmp r1 9
	jie sum_79

	cmp r1 0
	jie sum_70



next_8: 	cmp r1 1
	jie sum_81

	cmp r1 2
	jie sum_82

	cmp r1 3
	jie sum_83

	cmp r1 4
	jie sum_84

	cmp r1 5
	jie sum_85

	cmp r1 6
	jie sum_86

	cmp r1 7
	jie sum_87

	cmp r1 8
	jie sum_88

	cmp r1 9
	jie sum_89

	cmp r1 0
	jie sum_80



next_9: 	cmp r1 1
	jie sum_91

	cmp r1 2
	jie sum_92

	cmp r1 3
	jie sum_93

	cmp r1 4
	jie sum_94

	cmp r1 5
	jie sum_95

	cmp r1 6
	jie sum_96

	cmp r1 7
	jie sum_97

	cmp r1 8
	jie sum_98

	cmp r1 9
	jie sum_99

	cmp r1 0
	jie sum_90



next_0: 	cmp r1 1
	jie sum_01

	cmp r1 2
	jie sum_02

	cmp r1 3
	jie sum_03

	cmp r1 4
	jie sum_04

	cmp r1 5
	jie sum_05

	cmp r1 6
	jie sum_06

	cmp r1 7
	jie sum_07

	cmp r1 8
	jie sum_08

	cmp r1 9
	jie sum_09

	cmp r1 0
	jie sum_00




sum_11: cmp rc 1
	jie sum_11_1
	cmp rc 0
	jie sum_11_0

sum_11_1: mov r2 3
	mov rc 0
	ret

sum_11_0: mov r2 2
	mov rc 0
	ret


sum_12: cmp rc 1
	jie sum_12_1
	cmp rc 0
	jie sum_12_0

sum_12_1: mov r2 4
	mov rc 0
	ret

sum_12_0: mov r2 3
	mov rc 0
	ret


sum_13: cmp rc 1
	jie sum_13_1
	cmp rc 0
	jie sum_13_0

sum_13_1: mov r2 5
	mov rc 0
	ret

sum_13_0: mov r2 4
	mov rc 0
	ret


sum_14: cmp rc 1
	jie sum_14_1
	cmp rc 0
	jie sum_14_0

sum_14_1: mov r2 6
	mov rc 0
	ret

sum_14_0: mov r2 5
	mov rc 0
	ret


sum_15: cmp rc 1
	jie sum_15_1
	cmp rc 0
	jie sum_15_0

sum_15_1: mov r2 7
	mov rc 0
	ret

sum_15_0: mov r2 6
	mov rc 0
	ret


sum_16: cmp rc 1
	jie sum_16_1
	cmp rc 0
	jie sum_16_0

sum_16_1: mov r2 8
	mov rc 0
	ret

sum_16_0: mov r2 7
	mov rc 0
	ret


sum_17: cmp rc 1
	jie sum_17_1
	cmp rc 0
	jie sum_17_0

sum_17_1: mov r2 9
	mov rc 0
	ret

sum_17_0: mov r2 8
	mov rc 0
	ret


sum_18: cmp rc 1
	jie sum_18_1
	cmp rc 0
	jie sum_18_0

sum_18_1: mov r2 0
	mov rc 1
	ret

sum_18_0: mov r2 9
	mov rc 0
	ret


sum_19: cmp rc 1
	jie sum_19_1
	cmp rc 0
	jie sum_19_0

sum_19_1: mov r2 1
	mov rc 1
	ret

sum_19_0: mov r2 0
	mov rc 1
	ret


sum_10: cmp rc 1
	jie sum_10_1
	cmp rc 0
	jie sum_10_0

sum_10_1: mov r2 2
	mov rc 0
	ret

sum_10_0: mov r2 1
	mov rc 0
	ret


sum_21: cmp rc 1
	jie sum_21_1
	cmp rc 0
	jie sum_21_0

sum_21_1: mov r2 4
	mov rc 0
	ret

sum_21_0: mov r2 3
	mov rc 0
	ret


sum_22: cmp rc 1
	jie sum_22_1
	cmp rc 0
	jie sum_22_0

sum_22_1: mov r2 5
	mov rc 0
	ret

sum_22_0: mov r2 4
	mov rc 0
	ret


sum_23: cmp rc 1
	jie sum_23_1
	cmp rc 0
	jie sum_23_0

sum_23_1: mov r2 6
	mov rc 0
	ret

sum_23_0: mov r2 5
	mov rc 0
	ret


sum_24: cmp rc 1
	jie sum_24_1
	cmp rc 0
	jie sum_24_0

sum_24_1: mov r2 7
	mov rc 0
	ret

sum_24_0: mov r2 6
	mov rc 0
	ret


sum_25: cmp rc 1
	jie sum_25_1
	cmp rc 0
	jie sum_25_0

sum_25_1: mov r2 8
	mov rc 0
	ret

sum_25_0: mov r2 7
	mov rc 0
	ret


sum_26: cmp rc 1
	jie sum_26_1
	cmp rc 0
	jie sum_26_0

sum_26_1: mov r2 9
	mov rc 0
	ret

sum_26_0: mov r2 8
	mov rc 0
	ret


sum_27: cmp rc 1
	jie sum_27_1
	cmp rc 0
	jie sum_27_0

sum_27_1: mov r2 0
	mov rc 1
	ret

sum_27_0: mov r2 9
	mov rc 0
	ret


sum_28: cmp rc 1
	jie sum_28_1
	cmp rc 0
	jie sum_28_0

sum_28_1: mov r2 1
	mov rc 1
	ret

sum_28_0: mov r2 0
	mov rc 1
	ret


sum_29: cmp rc 1
	jie sum_29_1
	cmp rc 0
	jie sum_29_0

sum_29_1: mov r2 2
	mov rc 1
	ret

sum_29_0: mov r2 1
	mov rc 1
	ret


sum_20: cmp rc 1
	jie sum_20_1
	cmp rc 0
	jie sum_20_0

sum_20_1: mov r2 3
	mov rc 0
	ret

sum_20_0: mov r2 2
	mov rc 0
	ret


sum_31: cmp rc 1
	jie sum_31_1
	cmp rc 0
	jie sum_31_0

sum_31_1: mov r2 5
	mov rc 0
	ret

sum_31_0: mov r2 4
	mov rc 0
	ret


sum_32: cmp rc 1
	jie sum_32_1
	cmp rc 0
	jie sum_32_0

sum_32_1: mov r2 6
	mov rc 0
	ret

sum_32_0: mov r2 5
	mov rc 0
	ret


sum_33: cmp rc 1
	jie sum_33_1
	cmp rc 0
	jie sum_33_0

sum_33_1: mov r2 7
	mov rc 0
	ret

sum_33_0: mov r2 6
	mov rc 0
	ret


sum_34: cmp rc 1
	jie sum_34_1
	cmp rc 0
	jie sum_34_0

sum_34_1: mov r2 8
	mov rc 0
	ret

sum_34_0: mov r2 7
	mov rc 0
	ret


sum_35: cmp rc 1
	jie sum_35_1
	cmp rc 0
	jie sum_35_0

sum_35_1: mov r2 9
	mov rc 0
	ret

sum_35_0: mov r2 8
	mov rc 0
	ret


sum_36: cmp rc 1
	jie sum_36_1
	cmp rc 0
	jie sum_36_0

sum_36_1: mov r2 0
	mov rc 1
	ret

sum_36_0: mov r2 9
	mov rc 0
	ret


sum_37: cmp rc 1
	jie sum_37_1
	cmp rc 0
	jie sum_37_0

sum_37_1: mov r2 1
	mov rc 1
	ret

sum_37_0: mov r2 0
	mov rc 1
	ret


sum_38: cmp rc 1
	jie sum_38_1
	cmp rc 0
	jie sum_38_0

sum_38_1: mov r2 2
	mov rc 1
	ret

sum_38_0: mov r2 1
	mov rc 1
	ret


sum_39: cmp rc 1
	jie sum_39_1
	cmp rc 0
	jie sum_39_0

sum_39_1: mov r2 3
	mov rc 1
	ret

sum_39_0: mov r2 2
	mov rc 1
	ret


sum_30: cmp rc 1
	jie sum_30_1
	cmp rc 0
	jie sum_30_0

sum_30_1: mov r2 4
	mov rc 0
	ret

sum_30_0: mov r2 3
	mov rc 0
	ret


sum_41: cmp rc 1
	jie sum_41_1
	cmp rc 0
	jie sum_41_0

sum_41_1: mov r2 6
	mov rc 0
	ret

sum_41_0: mov r2 5
	mov rc 0
	ret


sum_42: cmp rc 1
	jie sum_42_1
	cmp rc 0
	jie sum_42_0

sum_42_1: mov r2 7
	mov rc 0
	ret

sum_42_0: mov r2 6
	mov rc 0
	ret


sum_43: cmp rc 1
	jie sum_43_1
	cmp rc 0
	jie sum_43_0

sum_43_1: mov r2 8
	mov rc 0
	ret

sum_43_0: mov r2 7
	mov rc 0
	ret


sum_44: cmp rc 1
	jie sum_44_1
	cmp rc 0
	jie sum_44_0

sum_44_1: mov r2 9
	mov rc 0
	ret

sum_44_0: mov r2 8
	mov rc 0
	ret


sum_45: cmp rc 1
	jie sum_45_1
	cmp rc 0
	jie sum_45_0

sum_45_1: mov r2 0
	mov rc 1
	ret

sum_45_0: mov r2 9
	mov rc 0
	ret


sum_46: cmp rc 1
	jie sum_46_1
	cmp rc 0
	jie sum_46_0

sum_46_1: mov r2 1
	mov rc 1
	ret

sum_46_0: mov r2 0
	mov rc 1
	ret


sum_47: cmp rc 1
	jie sum_47_1
	cmp rc 0
	jie sum_47_0

sum_47_1: mov r2 2
	mov rc 1
	ret

sum_47_0: mov r2 1
	mov rc 1
	ret


sum_48: cmp rc 1
	jie sum_48_1
	cmp rc 0
	jie sum_48_0

sum_48_1: mov r2 3
	mov rc 1
	ret

sum_48_0: mov r2 2
	mov rc 1
	ret


sum_49: cmp rc 1
	jie sum_49_1
	cmp rc 0
	jie sum_49_0

sum_49_1: mov r2 4
	mov rc 1
	ret

sum_49_0: mov r2 3
	mov rc 1
	ret


sum_40: cmp rc 1
	jie sum_40_1
	cmp rc 0
	jie sum_40_0

sum_40_1: mov r2 5
	mov rc 0
	ret

sum_40_0: mov r2 4
	mov rc 0
	ret


sum_51: cmp rc 1
	jie sum_51_1
	cmp rc 0
	jie sum_51_0

sum_51_1: mov r2 7
	mov rc 0
	ret

sum_51_0: mov r2 6
	mov rc 0
	ret


sum_52: cmp rc 1
	jie sum_52_1
	cmp rc 0
	jie sum_52_0

sum_52_1: mov r2 8
	mov rc 0
	ret

sum_52_0: mov r2 7
	mov rc 0
	ret


sum_53: cmp rc 1
	jie sum_53_1
	cmp rc 0
	jie sum_53_0

sum_53_1: mov r2 9
	mov rc 0
	ret

sum_53_0: mov r2 8
	mov rc 0
	ret


sum_54: cmp rc 1
	jie sum_54_1
	cmp rc 0
	jie sum_54_0

sum_54_1: mov r2 0
	mov rc 1
	ret

sum_54_0: mov r2 9
	mov rc 0
	ret


sum_55: cmp rc 1
	jie sum_55_1
	cmp rc 0
	jie sum_55_0

sum_55_1: mov r2 1
	mov rc 1
	ret

sum_55_0: mov r2 0
	mov rc 1
	ret


sum_56: cmp rc 1
	jie sum_56_1
	cmp rc 0
	jie sum_56_0

sum_56_1: mov r2 2
	mov rc 1
	ret

sum_56_0: mov r2 1
	mov rc 1
	ret


sum_57: cmp rc 1
	jie sum_57_1
	cmp rc 0
	jie sum_57_0

sum_57_1: mov r2 3
	mov rc 1
	ret

sum_57_0: mov r2 2
	mov rc 1
	ret


sum_58: cmp rc 1
	jie sum_58_1
	cmp rc 0
	jie sum_58_0

sum_58_1: mov r2 4
	mov rc 1
	ret

sum_58_0: mov r2 3
	mov rc 1
	ret


sum_59: cmp rc 1
	jie sum_59_1
	cmp rc 0
	jie sum_59_0

sum_59_1: mov r2 5
	mov rc 1
	ret

sum_59_0: mov r2 4
	mov rc 1
	ret


sum_50: cmp rc 1
	jie sum_50_1
	cmp rc 0
	jie sum_50_0

sum_50_1: mov r2 6
	mov rc 0
	ret

sum_50_0: mov r2 5
	mov rc 0
	ret


sum_61: cmp rc 1
	jie sum_61_1
	cmp rc 0
	jie sum_61_0

sum_61_1: mov r2 8
	mov rc 0
	ret

sum_61_0: mov r2 7
	mov rc 0
	ret


sum_62: cmp rc 1
	jie sum_62_1
	cmp rc 0
	jie sum_62_0

sum_62_1: mov r2 9
	mov rc 0
	ret

sum_62_0: mov r2 8
	mov rc 0
	ret


sum_63: cmp rc 1
	jie sum_63_1
	cmp rc 0
	jie sum_63_0

sum_63_1: mov r2 0
	mov rc 1
	ret

sum_63_0: mov r2 9
	mov rc 0
	ret


sum_64: cmp rc 1
	jie sum_64_1
	cmp rc 0
	jie sum_64_0

sum_64_1: mov r2 1
	mov rc 1
	ret

sum_64_0: mov r2 0
	mov rc 1
	ret


sum_65: cmp rc 1
	jie sum_65_1
	cmp rc 0
	jie sum_65_0

sum_65_1: mov r2 2
	mov rc 1
	ret

sum_65_0: mov r2 1
	mov rc 1
	ret


sum_66: cmp rc 1
	jie sum_66_1
	cmp rc 0
	jie sum_66_0

sum_66_1: mov r2 3
	mov rc 1
	ret

sum_66_0: mov r2 2
	mov rc 1
	ret


sum_67: cmp rc 1
	jie sum_67_1
	cmp rc 0
	jie sum_67_0

sum_67_1: mov r2 4
	mov rc 1
	ret

sum_67_0: mov r2 3
	mov rc 1
	ret


sum_68: cmp rc 1
	jie sum_68_1
	cmp rc 0
	jie sum_68_0

sum_68_1: mov r2 5
	mov rc 1
	ret

sum_68_0: mov r2 4
	mov rc 1
	ret


sum_69: cmp rc 1
	jie sum_69_1
	cmp rc 0
	jie sum_69_0

sum_69_1: mov r2 6
	mov rc 1
	ret

sum_69_0: mov r2 5
	mov rc 1
	ret


sum_60: cmp rc 1
	jie sum_60_1
	cmp rc 0
	jie sum_60_0

sum_60_1: mov r2 7
	mov rc 0
	ret

sum_60_0: mov r2 6
	mov rc 0
	ret


sum_71: cmp rc 1
	jie sum_71_1
	cmp rc 0
	jie sum_71_0

sum_71_1: mov r2 9
	mov rc 0
	ret

sum_71_0: mov r2 8
	mov rc 0
	ret


sum_72: cmp rc 1
	jie sum_72_1
	cmp rc 0
	jie sum_72_0

sum_72_1: mov r2 0
	mov rc 1
	ret

sum_72_0: mov r2 9
	mov rc 0
	ret


sum_73: cmp rc 1
	jie sum_73_1
	cmp rc 0
	jie sum_73_0

sum_73_1: mov r2 1
	mov rc 1
	ret

sum_73_0: mov r2 0
	mov rc 1
	ret


sum_74: cmp rc 1
	jie sum_74_1
	cmp rc 0
	jie sum_74_0

sum_74_1: mov r2 2
	mov rc 1
	ret

sum_74_0: mov r2 1
	mov rc 1
	ret


sum_75: cmp rc 1
	jie sum_75_1
	cmp rc 0
	jie sum_75_0

sum_75_1: mov r2 3
	mov rc 1
	ret

sum_75_0: mov r2 2
	mov rc 1
	ret


sum_76: cmp rc 1
	jie sum_76_1
	cmp rc 0
	jie sum_76_0

sum_76_1: mov r2 4
	mov rc 1
	ret

sum_76_0: mov r2 3
	mov rc 1
	ret


sum_77: cmp rc 1
	jie sum_77_1
	cmp rc 0
	jie sum_77_0

sum_77_1: mov r2 5
	mov rc 1
	ret

sum_77_0: mov r2 4
	mov rc 1
	ret


sum_78: cmp rc 1
	jie sum_78_1
	cmp rc 0
	jie sum_78_0

sum_78_1: mov r2 6
	mov rc 1
	ret

sum_78_0: mov r2 5
	mov rc 1
	ret


sum_79: cmp rc 1
	jie sum_79_1
	cmp rc 0
	jie sum_79_0

sum_79_1: mov r2 7
	mov rc 1
	ret

sum_79_0: mov r2 6
	mov rc 1
	ret


sum_70: cmp rc 1
	jie sum_70_1
	cmp rc 0
	jie sum_70_0

sum_70_1: mov r2 8
	mov rc 0
	ret

sum_70_0: mov r2 7
	mov rc 0
	ret


sum_81: cmp rc 1
	jie sum_81_1
	cmp rc 0
	jie sum_81_0

sum_81_1: mov r2 0
	mov rc 1
	ret

sum_81_0: mov r2 9
	mov rc 0
	ret


sum_82: cmp rc 1
	jie sum_82_1
	cmp rc 0
	jie sum_82_0

sum_82_1: mov r2 1
	mov rc 1
	ret

sum_82_0: mov r2 0
	mov rc 1
	ret


sum_83: cmp rc 1
	jie sum_83_1
	cmp rc 0
	jie sum_83_0

sum_83_1: mov r2 2
	mov rc 1
	ret

sum_83_0: mov r2 1
	mov rc 1
	ret


sum_84: cmp rc 1
	jie sum_84_1
	cmp rc 0
	jie sum_84_0

sum_84_1: mov r2 3
	mov rc 1
	ret

sum_84_0: mov r2 2
	mov rc 1
	ret


sum_85: cmp rc 1
	jie sum_85_1
	cmp rc 0
	jie sum_85_0

sum_85_1: mov r2 4
	mov rc 1
	ret

sum_85_0: mov r2 3
	mov rc 1
	ret


sum_86: cmp rc 1
	jie sum_86_1
	cmp rc 0
	jie sum_86_0

sum_86_1: mov r2 5
	mov rc 1
	ret

sum_86_0: mov r2 4
	mov rc 1
	ret


sum_87: cmp rc 1
	jie sum_87_1
	cmp rc 0
	jie sum_87_0

sum_87_1: mov r2 6
	mov rc 1
	ret

sum_87_0: mov r2 5
	mov rc 1
	ret


sum_88: cmp rc 1
	jie sum_88_1
	cmp rc 0
	jie sum_88_0

sum_88_1: mov r2 7
	mov rc 1
	ret

sum_88_0: mov r2 6
	mov rc 1
	ret


sum_89: cmp rc 1
	jie sum_89_1
	cmp rc 0
	jie sum_89_0

sum_89_1: mov r2 8
	mov rc 1
	ret

sum_89_0: mov r2 7
	mov rc 1
	ret


sum_80: cmp rc 1
	jie sum_80_1
	cmp rc 0
	jie sum_80_0

sum_80_1: mov r2 9
	mov rc 0
	ret

sum_80_0: mov r2 8
	mov rc 0
	ret


sum_91: cmp rc 1
	jie sum_91_1
	cmp rc 0
	jie sum_91_0

sum_91_1: mov r2 1
	mov rc 1
	ret

sum_91_0: mov r2 0
	mov rc 1
	ret


sum_92: cmp rc 1
	jie sum_92_1
	cmp rc 0
	jie sum_92_0

sum_92_1: mov r2 2
	mov rc 1
	ret

sum_92_0: mov r2 1
	mov rc 1
	ret


sum_93: cmp rc 1
	jie sum_93_1
	cmp rc 0
	jie sum_93_0

sum_93_1: mov r2 3
	mov rc 1
	ret

sum_93_0: mov r2 2
	mov rc 1
	ret


sum_94: cmp rc 1
	jie sum_94_1
	cmp rc 0
	jie sum_94_0

sum_94_1: mov r2 4
	mov rc 1
	ret

sum_94_0: mov r2 3
	mov rc 1
	ret


sum_95: cmp rc 1
	jie sum_95_1
	cmp rc 0
	jie sum_95_0

sum_95_1: mov r2 5
	mov rc 1
	ret

sum_95_0: mov r2 4
	mov rc 1
	ret


sum_96: cmp rc 1
	jie sum_96_1
	cmp rc 0
	jie sum_96_0

sum_96_1: mov r2 6
	mov rc 1
	ret

sum_96_0: mov r2 5
	mov rc 1
	ret


sum_97: cmp rc 1
	jie sum_97_1
	cmp rc 0
	jie sum_97_0

sum_97_1: mov r2 7
	mov rc 1
	ret

sum_97_0: mov r2 6
	mov rc 1
	ret


sum_98: cmp rc 1
	jie sum_98_1
	cmp rc 0
	jie sum_98_0

sum_98_1: mov r2 8
	mov rc 1
	ret

sum_98_0: mov r2 7
	mov rc 1
	ret


sum_99: cmp rc 1
	jie sum_99_1
	cmp rc 0
	jie sum_99_0

sum_99_1: mov r2 9
	mov rc 1
	ret

sum_99_0: mov r2 8
	mov rc 1
	ret


sum_90: cmp rc 1
	jie sum_90_1
	cmp rc 0
	jie sum_90_0

sum_90_1: mov r2 0
	mov rc 1
	ret

sum_90_0: mov r2 9
	mov rc 0
	ret


sum_01: cmp rc 1
	jie sum_01_1
	cmp rc 0
	jie sum_01_0

sum_01_1: mov r2 2
	mov rc 0
	ret

sum_01_0: mov r2 1
	mov rc 0
	ret


sum_02: cmp rc 1
	jie sum_02_1
	cmp rc 0
	jie sum_02_0

sum_02_1: mov r2 3
	mov rc 0
	ret

sum_02_0: mov r2 2
	mov rc 0
	ret


sum_03: cmp rc 1
	jie sum_03_1
	cmp rc 0
	jie sum_03_0

sum_03_1: mov r2 4
	mov rc 0
	ret

sum_03_0: mov r2 3
	mov rc 0
	ret


sum_04: cmp rc 1
	jie sum_04_1
	cmp rc 0
	jie sum_04_0

sum_04_1: mov r2 5
	mov rc 0
	ret

sum_04_0: mov r2 4
	mov rc 0
	ret


sum_05: cmp rc 1
	jie sum_05_1
	cmp rc 0
	jie sum_05_0

sum_05_1: mov r2 6
	mov rc 0
	ret

sum_05_0: mov r2 5
	mov rc 0
	ret


sum_06: cmp rc 1
	jie sum_06_1
	cmp rc 0
	jie sum_06_0

sum_06_1: mov r2 7
	mov rc 0
	ret

sum_06_0: mov r2 6
	mov rc 0
	ret


sum_07: cmp rc 1
	jie sum_07_1
	cmp rc 0
	jie sum_07_0

sum_07_1: mov r2 8
	mov rc 0
	ret

sum_07_0: mov r2 7
	mov rc 0
	ret


sum_08: cmp rc 1
	jie sum_08_1
	cmp rc 0
	jie sum_08_0

sum_08_1: mov r2 9
	mov rc 0
	ret

sum_08_0: mov r2 8
	mov rc 0
	ret


sum_09: cmp rc 1
	jie sum_09_1
	cmp rc 0
	jie sum_09_0

sum_09_1: mov r2 0
	mov rc 1
	ret

sum_09_0: mov r2 9
	mov rc 0
	ret


sum_00: cmp rc 1
	jie sum_00_1
	cmp rc 0
	jie sum_00_0

sum_00_1: mov r2 1
	mov rc 0
	ret

sum_00_0: mov r2 0
	mov rc 0
	ret


"""}

from nvm import make_scaled_nvm


my_nvm = make_scaled_nvm(
     register_names = register_names,
     programs = programs,
     orthogonal=True)
	 
my_nvm.assemble(programs,verbose=2)

my_nvm.initialize_memory(pointers={"0": {"r3": "A"} , "4": {"r4": "B"} , "6": {"r5": "R"}},
values={"0": {"r0": "8"} , "1": {"r0": "/"} ,"2": {"r0": "/"} ,"3": {"r0": "/"},"4": {"r1": "9"},"5": {"r1": "/"},"6": {"r2": "/"} })

my_nvm.load("myfirstprogram",
     initial_state = {"rc":"0"})
	 
import itertools

print(my_nvm.net.layers["r0"].size)
print(my_nvm.net.layers["r0"].coder.encodings.keys())



for t in itertools.count():
     #my_nvm.net.tick()
     my_nvm.step()
     if my_nvm.at_exit(): break

t

my_nvm.decode_state(layer_names=register_names)
