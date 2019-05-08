### How to add 2 numbers in NVM?
1. To use the addition feature, open the file addition.py in src/nvm.
2. Initialize the memory with proper memory locations for first number, second number, result in r0, r3 and r5 registers respectively.
3. Initialize the values in memory locations for each register r0 and r1 representing first and second number in reverse. Store the digits of first and second number in contiguous memory locations each. The starting memory locations of both the numbers and the result need not be contiguous.
4. Initialize register rc = 0 as default carry before addition starts.
5. Execute this file from console. 
6. The answer of addition will be stored in the memory location provided. The pointer to the answer will be at the last digit of the result. The result is also stored in reverse from the starting memory location.

The files addition_lookup.py is the utility code to generate addition_lookup.txt for reference. The lookup data is used separately in the addition.py file as part of the input program loaded.
