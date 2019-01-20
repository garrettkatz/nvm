nback_programs = {
"nback":"""

            jmp start{n}

    start2: mov tc ol
            mem tc
            nxt
            mov mc no
            mov mc hold
    start1: mov tc ol
            mem tc
            nxt
            mov mc no
            mov mc hold

    repeat: mov tc ol
            mem tc
            jmp back{n}

    back2:  prv
    back1:  prv
            rem tc
            cmp tc ol
            jie match

            mov mc no
            jmp hold
    match:  mov mc yes
    hold:   mov mc hold
            jmp forw{n}
            
    forw2:  nxt
    forw1:  nxt
            nxt
            jmp repeat

            exit
    
""".format(n=1)
}

