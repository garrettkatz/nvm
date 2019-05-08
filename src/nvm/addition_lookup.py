list=["1","2","3","4","5","6","7","8","9","0"]

starting="programs={\n\"myfirstprogram\":\"\"\""

ending="\n\n\"\"\"}"

enter="\n\n\tnop\n\tsub add\n\texit"

start="\n\nadd: "

first=""

for l in list:
    temp="\tcmp r0 "+l+"\n\tjie next_"+l+"\n\n"
    start=start+temp
first=start


second=""
mid_start=""
for l in list:
    mid_start=mid_start+"\n\nnext_"+l+": "
    for i in list:
        temp="\tcmp r1 "+i+"\n\tjie sum_"+l+i+"\n\n"
        mid_start=mid_start+temp
second=mid_start



third=""
temp=""
for l in list:
    for i in list:
        # temp=temp+"\n\n\nsum_"+l+i+": "+"cmp rc 1"+"\n\tjie sum_"+l+i+"_1"+"\n\tsum_"+l+i+"_0"
        temp=temp+"\n\n\nsum_"+l+i+": "+"cmp rc 1"+"\n\tjie sum_"+l+i+"_1"+"\n\tcmp rc 0"+"\n\tjie sum_"+l+i+"_0"
        temp=temp+"\n\nsum_"+l+i+"_1"+": "+"mov r2 "+str((int(l)+int(i)+1)%10)+"\n\tmov rc "+str((int(l)+int(i)+1)//10)+"\n\tret"
        temp=temp+"\n\nsum_"+l+i+"_0"+": "+"mov r2 "+str((int(l)+int(i)+0)%10)+"\n\tmov rc "+str((int(l)+int(i)+0)//10)+"\n\tret"
third=temp

final=starting+enter+first+second+third+ending

with open("addition_output.txt", "a") as text_file:
    print(final, file=text_file)





