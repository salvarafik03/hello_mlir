add_mlir_dialect_library(MLIRMelo
		MeloDialect.cpp
		MeloOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Melo

        DEPENDS
        MLIRMeloOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
