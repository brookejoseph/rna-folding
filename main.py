from rna_folding import RNA3DFolding


rna_folder = RNA3DFolding()
result = rna_folder.run_convolution()
print(f"Output shape: {result.shape}")
print(result)
