import selfies as sf
smiles1 = ['c1cc(ccc1N)S(=O)(=O)Nc2cncc(n2)Cl', 'c1cc(ccc1N)S(=O)(=O)Nc2ccc(nn2)Cl', 'c1cc(ccc1N)S(=O)(=O)NC(=N)N', 'Cc1ccnc(n1)NS(=O)(=O)c2ccc(cc2)N', 'Cc1nnc(s1)NS(=O)(=O)c2ccc(cc2)N', 'Cc1c(oc(n1)NS(=O)(=O)c2ccc(cc2)N)C', 'c1cc(ccc1N)S(=O)(=O)N', 'CCN(CC)CCSCC(=O)O[C@@H]1C[C@@]([C@H]([C@@H]([C@@]23CC[C@H]([C@@]1([C@@H]2C(=O)CC3)C)C)C)O)(C)C=C', 'CCCS(=O)c1ccc2c(c1)nc([nH]2)NC(=O)OC', 'c1cc(c(c(c1)Cl)Cn2cnc3c2ncnc3N)F', 'Cc1c(c(c(c(n1)C)Cl)O)Cl', 'CCOc1cc(ccc1C(=O)OC)NC(=O)C', 'CCN(CCCC(=O)N1CCC(CC1)C(=O)N(C)C)[C@@H](C)CC2=CC=C(C=C2)OC', 'C1=CC(=CC=C1/C=N/NC(=N)N/N=C/C2=CC=C(C=C2)Cl)Cl', 'CC(C)(C)NCC(c1cc(c(c(c1)Cl)N)C(F)(F)F)O']
smiles2 = smiles1.copy()
selfies1, selfies2 = list(map(sf.encoder, smiles1)), list(map(sf.encoder, smiles2))
print(selfies1[:3], '\n', selfies2[:3])
print(selfies1==selfies2)
selfies1, selfies2 = [list(sf.split_selfies(selfie)) for selfie in selfies1], [list(sf.split_selfies(selfie)) for selfie in selfies2]
print(list(selfies1[0]), '\n', list(selfies2[0]))
print(selfies1==selfies2)