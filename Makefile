SHELL := /bin/bash


projects = dev prod

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
DIR := $(dir $(mkfile_path))

IS_MAKEFILE_RUNNING_TARGETS ?= 1



## Check if make command is in a project. 
## If so it will set env var that makefile is running. 
## Then it calls the project default rule to set environment variables for the project 
## before calling the command passed in next. 
## ex. `make dev build`

define PROGRAM_projects
ifneq ($$(filter $(firstword $(MAKECMDGOALS)), $(1)),)
$$(info exist $$(filter $(firstword $(MAKECMDGOALS)),$($1)))
$$(eval TARGETPROJ=$1)
RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$$(eval $(RUN_ARGS):;@:)
$$(eval IS_MAKEFILE_RUNNING_TARGETS=0)
TARGETDIR := ${DIR}$@
else
# $$(info $(firstword $(MAKECMDGOALS)) does not exist in ${1})
endif
endef


## Set env var for the project.

$(projects):	
	$(eval export IMAGE=us.gcr.io/kapwing-$@/mediapipe)

## Calls the target after the env for the project have been set 

define DEFAULTTARGET
	@echo "DEFAULTTARGET, ${1}, ${TARGETDIR}, ${RUN_ARGS},"
	$(MAKE) -C ${TARGETDIR} -f ${mkfile_path} ${RUN_ARGS} 

	@echo ""
	@echo 'Running something after all rules finished'
endef


$(foreach proj,$(projects),$(eval $(call PROGRAM_projects,$(proj))))



ifeq (${IS_MAKEFILE_RUNNING_TARGETS},0)

# $(info "inside running target ${IS_MAKEFILE_RUNNING_TARGETS}")

%:
	@:
		# @echo "here $@"
		# @echo "IS_MAKEFILE_RUNNING_TARGETS= ${IS_MAKEFILE_RUNNING_TARGETS}, $(TARGETDIR)"		

		$(if $(patsubst 0,,${IS_MAKEFILE_RUNNING_TARGETS}),,$(call DEFAULTTARGET))

else

build:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.kapwing --cache-from $(IMAGE):latest -t $(IMAGE):latest --build-arg BUILDKIT_INLINE_CACHE=1 .

push:
	docker push $(IMAGE):latest	

testit:
	@echo "IMAGE= $(IMAGE)"
	@echo "MK: ${IS_MAKEFILE_RUNNING_TARGETS}"
	@ls | wc -l


endif