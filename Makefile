# target
TARGET = stereo 

# shell command
CXX = g++
RM = rm -fr

# directory
OUTDIR = build
SRCDIR = src
SUBDIRS = $(shell find $(SRCDIR) -type d)

INCDIR = -Isrc \
	-Isrc/include \
	-Isrc/sparse_stereo \
	-I/usr/local/cuda/include \
	`pkg-config --cflags opencv`

LIBDIR = `pkg-config --libs-only-L opencv`

LIBS = `pkg-config --libs-only-l opencv`

# cpp source file
SRCCPP = $(foreach dir,$(SUBDIRS),$(wildcard $(dir)/*.cpp))
CPPOBJS = $(foreach file, $(SRCCPP:.cpp=.cpp.o), $(OUTDIR)/$(file))
CPPDEP = $(foreach file, $(SRCCPP:.cpp=.cpp.d), $(OUTDIR)/$(file))

# object file
OBJS = $(CPPOBJS)
DEPENDFILES = $(CPPDEP)

CXXFLAGS = $(INCDIR) -fpermissive -std=c++11 -O3
LDFLAGS = $(LIBDIR) $(LIBS) -fopenmp

# defination
.SUFFIXES: .cpp .h .d
.PHONY: mk_dir clean all echo

# rules
all: $(TARGET)

mk_dir:
	@[ -d $(OUTDIR) ] || mkdir -p $(OUTDIR); \
	for val in $(SUBDIRS); do \
		[ -d $(OUTDIR)/$${val} ] || mkdir -p  $(OUTDIR)/$${val};\
	done;

echo:
	@echo 'SUBDIRS:$(SUBDIRS)'
	@echo 'CXXFLAGS:$(CXXFLAGS)'
	@echo 'OBJS:$(OBJS)'
	@echo 'DEPENDFILES:$(DEPENDFILES)'
	@echo 'LDFLAGS:$(LDFLAGS)'

$(OUTDIR)/%.cpp.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET):$(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	-@ $(RM) $(OUTDIR)/* $(TARGET)

# source and header file dependent
-include $(DEPENDFILES)
$(OUTDIR)/%.cpp.d:%.cpp | mk_dir
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$
