��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.12v2.15.0-11-g63f5a65c7cd8��
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �@
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  @�
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  @@
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   �
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  @@
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ff�@
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   �
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *   �
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *  JC
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *   @
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *   �
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *   �
M
Const_16Const*
_output_shapes
: *
dtype0*
valueB
 *  D
M
Const_17Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
M
Const_18Const*
_output_shapes
: *
dtype0*
valueB
 *  HC
M
Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
M
Const_20Const*
_output_shapes
: *
dtype0*
valueB
 *  �@
M
Const_21Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
M
Const_22Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
M
Const_23Const*
_output_shapes
: *
dtype0*
valueB
 *   �
M
Const_24Const*
_output_shapes
: *
dtype0*
valueB
 *  �B
M
Const_25Const*
_output_shapes
: *
dtype0*
valueB
 *  ��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_2/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_2/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_2/kernel/*
dtype0*
shape
: *&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_2/kernel/*
dtype0*
shape
: *&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes

: *
dtype0
�
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_1/bias/*
dtype0*
shape: *$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_1/bias/*
dtype0*
shape: *$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_1/kernel/*
dtype0*
shape
:@ *&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes

:@ *
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_1/kernel/*
dtype0*
shape
:@ *&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes

:@ *
dtype0
�
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/v/dense/bias/*
dtype0*
shape:@*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/m/dense/bias/*
dtype0*
shape:@*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense/kernel/*
dtype0*
shape
:@*$
shared_nameAdam/v/dense/kernel
{
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense/kernel/*
dtype0*
shape
:@*$
shared_nameAdam/m/dense/kernel
{
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes

:@*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesConst_25Const_24Const_23Const_22Const_21Const_20Const_19Const_18Const_17Const_16Const_15Const_14Const_13Const_12Const_11Const_10Const_9Const_8Const_7Const_6Const_5Const_4Const_3Const_2Const_1Constdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1445843

NoOpNoOp
�T
Const_26Const"/device:CPU:0*
_output_shapes
: *
dtype0*�S
value�SB�S B�S
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-0
layer-14
layer-15
layer_with_weights-1
layer-16
layer-17
layer_with_weights-2
layer-18
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator* 
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
$P _saved_model_loader_tracked_dict* 
.
*0
+1
92
:3
H4
I5*
.
*0
+1
92
:3
H4
I5*

Q0
R1* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Xtrace_0
Ytrace_1* 

Ztrace_0
[trace_1* 
* 
�
\
_variables
]_iterations
^_learning_rate
__index_dict
`
_momentums
a_velocities
b_update_step_xla*

cserving_default* 
* 
* 
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

itrace_0* 

jtrace_0* 

*0
+1*

*0
+1*
	
Q0* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

wtrace_0
xtrace_1* 

ytrace_0
ztrace_1* 
* 

90
:1*

90
:1*
	
R0* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

H0
I1*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
y
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map* 

�trace_0* 

�trace_0* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*

�0
�1*
* 
* 
* 
* 
* 
* 
n
]0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
Q0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
R0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25* 
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25* 
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn* 
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�serving_default* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotal_1count_1totalcountConst_26*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1446768
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1446849��

�
d
+__inference_dropout_1_layer_call_fn_1446518

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446535

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
)__inference_dropout_layer_call_fn_1446467

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1446151o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�.
�
:__inference_transform_features_layer_layer_call_fn_1446086
age	
bp	
placeholder	
cholesterol	
placeholder_1	
placeholder_2	
placeholder_3	
placeholder_4	
placeholder_5	
placeholder_6
sex	
placeholder_7	
thallium	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12�
PartitionedCallPartitionedCallagebpplaceholdercholesterolplaceholder_1placeholder_2placeholder_3placeholder_4placeholder_5placeholder_6sexplaceholder_7thalliumunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*2
Tin+
)2'												*
Tout
2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_transform_features_layer_layer_call_and_return_conditional_losses_1445993`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:���������b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:���������b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:���������b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:���������b

Identity_7IdentityPartitionedCall:output:7*
T0*'
_output_shapes
:���������b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:���������b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:���������d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:���������d
Identity_11IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:���������d
Identity_12IdentityPartitionedCall:output:12*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : :L H
'
_output_shapes
:���������

_user_specified_nameAge:KG
'
_output_shapes
:���������

_user_specified_nameBP:XT
'
_output_shapes
:���������
)
_user_specified_nameChest pain type:TP
'
_output_shapes
:���������
%
_user_specified_nameCholesterol:TP
'
_output_shapes
:���������
%
_user_specified_nameEKG results:XT
'
_output_shapes
:���������
)
_user_specified_nameExercise angina:UQ
'
_output_shapes
:���������
&
_user_specified_nameFBS over 120:OK
'
_output_shapes
:���������
 
_user_specified_nameMax HR:`\
'
_output_shapes
:���������
1
_user_specified_nameNumber of vessels fluro:V	R
'
_output_shapes
:���������
'
_user_specified_nameST depression:L
H
'
_output_shapes
:���������

_user_specified_nameSex:TP
'
_output_shapes
:���������
%
_user_specified_nameSlope of ST:QM
'
_output_shapes
:���������
"
_user_specified_name
Thallium:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: 
�

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446184

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

c
D__inference_dropout_layer_call_and_return_conditional_losses_1446484

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

c
D__inference_dropout_layer_call_and_return_conditional_losses_1446151

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_dense_layer_call_and_return_conditional_losses_1446134

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�6
�
B__inference_model_layer_call_and_return_conditional_losses_1446211

age_xf

sex_xf
placeholder	
bp_xf
cholesterol_xf
placeholder_1
placeholder_2
placeholder_3
placeholder_4
placeholder_5
placeholder_6
placeholder_7
thallium_xf
dense_1446135:@
dense_1446137:@!
dense_1_1446168:@ 
dense_1_1446170: !
dense_2_1446197: 
dense_2_1446199:
identity��dense/StatefulPartitionedCall�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCallage_xfsex_xfplaceholderbp_xfcholesterol_xfplaceholder_1placeholder_2placeholder_3placeholder_4placeholder_5placeholder_6placeholder_7thallium_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1446118�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1446135dense_1446137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1446134�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1446151�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_1446168dense_1_1446170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1446167�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446184�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_1446197dense_2_1446199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1446196|
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1446135*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_1446168*
_output_shapes

:@ *
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameAge_xf:OK
'
_output_shapes
:���������
 
_user_specified_nameSex_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameChest pain type_xf:NJ
'
_output_shapes
:���������

_user_specified_nameBP_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameCholesterol_xf:XT
'
_output_shapes
:���������
)
_user_specified_nameFBS over 120_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameEKG results_xf:RN
'
_output_shapes
:���������
#
_user_specified_name	Max HR_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameExercise angina_xf:Y	U
'
_output_shapes
:���������
*
_user_specified_nameST depression_xf:W
S
'
_output_shapes
:���������
(
_user_specified_nameSlope of ST_xf:c_
'
_output_shapes
:���������
4
_user_specified_nameNumber of vessels fluro_xf:TP
'
_output_shapes
:���������
%
_user_specified_nameThallium_xf:'#
!
_user_specified_name	1446135:'#
!
_user_specified_name	1446137:'#
!
_user_specified_name	1446168:'#
!
_user_specified_name	1446170:'#
!
_user_specified_name	1446197:'#
!
_user_specified_name	1446199
�0
�
"__inference__wrapped_model_1445884

age_xf

sex_xf
placeholder	
bp_xf
cholesterol_xf
placeholder_1
placeholder_2
placeholder_3
placeholder_4
placeholder_5
placeholder_6
placeholder_7
thallium_xf<
*model_dense_matmul_readvariableop_resource:@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2age_xfsex_xfplaceholderbp_xfcholesterol_xfplaceholder_1placeholder_2placeholder_3placeholder_4placeholder_5placeholder_6placeholder_7thallium_xf&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@t
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:���������@�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� x
model/dropout_1/IdentityIdentity model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:��������� �
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model/dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitymodel/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameAge_xf:OK
'
_output_shapes
:���������
 
_user_specified_nameSex_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameChest pain type_xf:NJ
'
_output_shapes
:���������

_user_specified_nameBP_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameCholesterol_xf:XT
'
_output_shapes
:���������
)
_user_specified_nameFBS over 120_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameEKG results_xf:RN
'
_output_shapes
:���������
#
_user_specified_name	Max HR_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameExercise angina_xf:Y	U
'
_output_shapes
:���������
*
_user_specified_nameST depression_xf:W
S
'
_output_shapes
:���������
(
_user_specified_nameSlope of ST_xf:c_
'
_output_shapes
:���������
4
_user_specified_nameNumber of vessels fluro_xf:TP
'
_output_shapes
:���������
%
_user_specified_nameThallium_xf:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446540

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1446236

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
 __inference__traced_save_1446768
file_prefix5
#read_disablecopyonread_dense_kernel:@1
#read_1_disablecopyonread_dense_bias:@9
'read_2_disablecopyonread_dense_1_kernel:@ 3
%read_3_disablecopyonread_dense_1_bias: 9
'read_4_disablecopyonread_dense_2_kernel: 3
%read_5_disablecopyonread_dense_2_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: >
,read_8_disablecopyonread_adam_m_dense_kernel:@>
,read_9_disablecopyonread_adam_v_dense_kernel:@9
+read_10_disablecopyonread_adam_m_dense_bias:@9
+read_11_disablecopyonread_adam_v_dense_bias:@A
/read_12_disablecopyonread_adam_m_dense_1_kernel:@ A
/read_13_disablecopyonread_adam_v_dense_1_kernel:@ ;
-read_14_disablecopyonread_adam_m_dense_1_bias: ;
-read_15_disablecopyonread_adam_v_dense_1_bias: A
/read_16_disablecopyonread_adam_m_dense_2_kernel: A
/read_17_disablecopyonread_adam_v_dense_2_kernel: ;
-read_18_disablecopyonread_adam_m_dense_2_bias:;
-read_19_disablecopyonread_adam_v_dense_2_bias:+
!read_20_disablecopyonread_total_1: +
!read_21_disablecopyonread_count_1: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const_26
identity_49��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@ y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

: y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_adam_m_dense_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_9/DisableCopyOnReadDisableCopyOnRead,read_9_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp,read_9_disablecopyonread_adam_v_dense_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_10/DisableCopyOnReadDisableCopyOnRead+read_10_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp+read_10_disablecopyonread_adam_m_dense_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_adam_v_dense_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_12/DisableCopyOnReadDisableCopyOnRead/read_12_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp/read_12_disablecopyonread_adam_m_dense_1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_adam_v_dense_1_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_14/DisableCopyOnReadDisableCopyOnRead-read_14_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp-read_14_disablecopyonread_adam_m_dense_1_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_adam_v_dense_1_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_adam_m_dense_2_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_v_dense_2_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_adam_m_dense_2_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead-read_19_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp-read_19_disablecopyonread_adam_v_dense_2_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_total_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_count_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const_26"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_49Identity_49:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_user_specified_namedense/kernel:*&
$
_user_specified_name
dense/bias:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_2/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:3	/
-
_user_specified_nameAdam/m/dense/kernel:3
/
-
_user_specified_nameAdam/v/dense/kernel:1-
+
_user_specified_nameAdam/m/dense/bias:1-
+
_user_specified_nameAdam/v/dense/bias:51
/
_user_specified_nameAdam/m/dense_1/kernel:51
/
_user_specified_nameAdam/v/dense_1/kernel:3/
-
_user_specified_nameAdam/m/dense_1/bias:3/
-
_user_specified_nameAdam/v/dense_1/bias:51
/
_user_specified_nameAdam/m/dense_2/kernel:51
/
_user_specified_nameAdam/v/dense_2/kernel:3/
-
_user_specified_nameAdam/m/dense_2/bias:3/
-
_user_specified_nameAdam/v/dense_2/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:@<

_output_shapes
: 
"
_user_specified_name
Const_26
�
�
'__inference_model_layer_call_fn_1446292

age_xf

sex_xf
placeholder	
bp_xf
cholesterol_xf
placeholder_1
placeholder_2
placeholder_3
placeholder_4
placeholder_5
placeholder_6
placeholder_7
thallium_xf
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallage_xfsex_xfplaceholderbp_xfcholesterol_xfplaceholder_1placeholder_2placeholder_3placeholder_4placeholder_5placeholder_6placeholder_7thallium_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1446211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameAge_xf:OK
'
_output_shapes
:���������
 
_user_specified_nameSex_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameChest pain type_xf:NJ
'
_output_shapes
:���������

_user_specified_nameBP_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameCholesterol_xf:XT
'
_output_shapes
:���������
)
_user_specified_nameFBS over 120_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameEKG results_xf:RN
'
_output_shapes
:���������
#
_user_specified_name	Max HR_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameExercise angina_xf:Y	U
'
_output_shapes
:���������
*
_user_specified_nameST depression_xf:W
S
'
_output_shapes
:���������
(
_user_specified_nameSlope of ST_xf:c_
'
_output_shapes
:���������
4
_user_specified_nameNumber of vessels fluro_xf:TP
'
_output_shapes
:���������
%
_user_specified_nameThallium_xf:'#
!
_user_specified_name	1446278:'#
!
_user_specified_name	1446280:'#
!
_user_specified_name	1446282:'#
!
_user_specified_name	1446284:'#
!
_user_specified_name	1446286:'#
!
_user_specified_name	1446288
�
�
'__inference_dense_layer_call_fn_1446447

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1446134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1446441:'#
!
_user_specified_name	1446443
�	
�
__inference_loss_fn_1_1446576K
9dense_1_kernel_regularizer_l2loss_readvariableop_resource:@ 
identity��0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@ *
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: U
NoOpNoOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
__inference_loss_fn_0_1446568I
7dense_kernel_regularizer_l2loss_readvariableop_resource:@
identity��.dense/kernel/Regularizer/L2Loss/ReadVariableOp�
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
G
+__inference_dropout_1_layer_call_fn_1446523

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446247`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446247

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
-__inference_concatenate_layer_call_fn_1446420
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1446118`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_12
�
�
H__inference_concatenate_layer_call_and_return_conditional_losses_1446118

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O	K
'
_output_shapes
:���������
 
_user_specified_nameinputs:O
K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_1446513

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�/
�
%__inference_signature_wrapper_1445527

inputs	
inputs_1	
	inputs_10
	inputs_11	
	inputs_12	
	inputs_13	
inputs_2	
inputs_3	
inputs_4	
inputs_5	
inputs_6	
inputs_7	
inputs_8	
inputs_9	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7	

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13�
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*3
Tin,
*2(													*
Tout
2	*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_pruned_1445457`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:���������b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:���������b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:���������b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:���������b

Identity_7IdentityPartitionedCall:output:7*
T0	*'
_output_shapes
:���������b

Identity_8IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:���������b

Identity_9IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:���������d
Identity_10IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:���������d
Identity_11IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:���������d
Identity_12IdentityPartitionedCall:output:12*
T0*'
_output_shapes
:���������d
Identity_13IdentityPartitionedCall:output:13*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : :O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_12:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_13:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:Q
M
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_9:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
�q
�
(__inference_serve_tf_examples_fn_1445773
examples$
 transform_features_layer_1445681$
 transform_features_layer_1445683$
 transform_features_layer_1445685$
 transform_features_layer_1445687$
 transform_features_layer_1445689$
 transform_features_layer_1445691$
 transform_features_layer_1445693$
 transform_features_layer_1445695$
 transform_features_layer_1445697$
 transform_features_layer_1445699$
 transform_features_layer_1445701$
 transform_features_layer_1445703$
 transform_features_layer_1445705$
 transform_features_layer_1445707$
 transform_features_layer_1445709$
 transform_features_layer_1445711$
 transform_features_layer_1445713$
 transform_features_layer_1445715$
 transform_features_layer_1445717$
 transform_features_layer_1445719$
 transform_features_layer_1445721$
 transform_features_layer_1445723$
 transform_features_layer_1445725$
 transform_features_layer_1445727$
 transform_features_layer_1445729$
 transform_features_layer_1445731<
*model_dense_matmul_readvariableop_resource:@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOpU
ParseExample/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_5Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_6Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_7Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_8Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_9Const*
_output_shapes
: *
dtype0*
valueB X
ParseExample/Const_10Const*
_output_shapes
: *
dtype0	*
valueB	 X
ParseExample/Const_11Const*
_output_shapes
: *
dtype0	*
valueB	 X
ParseExample/Const_12Const*
_output_shapes
: *
dtype0	*
valueB	 d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB �
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*�
value�B�BAgeBBPBChest pain typeBCholesterolBEKG resultsBExercise anginaBFBS over 120BMax HRBNumber of vessels fluroBST depressionBSexBSlope of STBThalliumj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0ParseExample/Const_10:output:0ParseExample/Const_11:output:0ParseExample/Const_12:output:0*
Tdense
2												*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*`
dense_shapesP
N:::::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 �
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0	*
_output_shapes
::��v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0	*
_output_shapes
::��x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:����������
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
(transform_features_layer/PartitionedCallPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:2*ParseExample/ParseExampleV2:dense_values:3*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:68transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9+ParseExample/ParseExampleV2:dense_values:10+ParseExample/ParseExampleV2:dense_values:11+ParseExample/ParseExampleV2:dense_values:12 transform_features_layer_1445681 transform_features_layer_1445683 transform_features_layer_1445685 transform_features_layer_1445687 transform_features_layer_1445689 transform_features_layer_1445691 transform_features_layer_1445693 transform_features_layer_1445695 transform_features_layer_1445697 transform_features_layer_1445699 transform_features_layer_1445701 transform_features_layer_1445703 transform_features_layer_1445705 transform_features_layer_1445707 transform_features_layer_1445709 transform_features_layer_1445711 transform_features_layer_1445713 transform_features_layer_1445715 transform_features_layer_1445717 transform_features_layer_1445719 transform_features_layer_1445721 transform_features_layer_1445723 transform_features_layer_1445725 transform_features_layer_1445727 transform_features_layer_1445729 transform_features_layer_1445731*3
Tin,
*2(													*
Tout
2	*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_pruned_1445457_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV21transform_features_layer/PartitionedCall:output:02transform_features_layer/PartitionedCall:output:111transform_features_layer/PartitionedCall:output:21transform_features_layer/PartitionedCall:output:11transform_features_layer/PartitionedCall:output:31transform_features_layer/PartitionedCall:output:61transform_features_layer/PartitionedCall:output:41transform_features_layer/PartitionedCall:output:81transform_features_layer/PartitionedCall:output:52transform_features_layer/PartitionedCall:output:102transform_features_layer/PartitionedCall:output:121transform_features_layer/PartitionedCall:output:92transform_features_layer/PartitionedCall:output:13&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@t
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:���������@�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� x
model/dropout_1/IdentityIdentity model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:��������� �
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model/dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitymodel/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource
�o
�
#__inference__traced_restore_1446849
file_prefix/
assignvariableop_dense_kernel:@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@ -
assignvariableop_3_dense_1_bias: 3
!assignvariableop_4_dense_2_kernel: -
assignvariableop_5_dense_2_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: 8
&assignvariableop_8_adam_m_dense_kernel:@8
&assignvariableop_9_adam_v_dense_kernel:@3
%assignvariableop_10_adam_m_dense_bias:@3
%assignvariableop_11_adam_v_dense_bias:@;
)assignvariableop_12_adam_m_dense_1_kernel:@ ;
)assignvariableop_13_adam_v_dense_1_kernel:@ 5
'assignvariableop_14_adam_m_dense_1_bias: 5
'assignvariableop_15_adam_v_dense_1_bias: ;
)assignvariableop_16_adam_m_dense_2_kernel: ;
)assignvariableop_17_adam_v_dense_2_kernel: 5
'assignvariableop_18_adam_m_dense_2_bias:5
'assignvariableop_19_adam_v_dense_2_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_adam_m_dense_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_adam_v_dense_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp%assignvariableop_10_adam_m_dense_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_adam_v_dense_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_dense_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_dense_1_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_m_dense_1_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_v_dense_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_dense_2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_dense_2_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_2_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_25Identity_25:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_user_specified_namedense/kernel:*&
$
_user_specified_name
dense/bias:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_2/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:3	/
-
_user_specified_nameAdam/m/dense/kernel:3
/
-
_user_specified_nameAdam/v/dense/kernel:1-
+
_user_specified_nameAdam/m/dense/bias:1-
+
_user_specified_nameAdam/v/dense/bias:51
/
_user_specified_nameAdam/m/dense_1/kernel:51
/
_user_specified_nameAdam/v/dense_1/kernel:3/
-
_user_specified_nameAdam/m/dense_1/bias:3/
-
_user_specified_nameAdam/v/dense_1/bias:51
/
_user_specified_nameAdam/m/dense_2/kernel:51
/
_user_specified_nameAdam/v/dense_2/kernel:3/
-
_user_specified_nameAdam/m/dense_2/bias:3/
-
_user_specified_nameAdam/v/dense_2/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount
�
E
)__inference_dropout_layer_call_fn_1446472

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1446236`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_1446196

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1446489

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�3
�
B__inference_model_layer_call_and_return_conditional_losses_1446263

age_xf

sex_xf
placeholder	
bp_xf
cholesterol_xf
placeholder_1
placeholder_2
placeholder_3
placeholder_4
placeholder_5
placeholder_6
placeholder_7
thallium_xf
dense_1446227:@
dense_1446229:@!
dense_1_1446238:@ 
dense_1_1446240: !
dense_2_1446249: 
dense_2_1446251:
identity��dense/StatefulPartitionedCall�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�dense_2/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCallage_xfsex_xfplaceholderbp_xfcholesterol_xfplaceholder_1placeholder_2placeholder_3placeholder_4placeholder_5placeholder_6placeholder_7thallium_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1446118�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1446227dense_1446229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1446134�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1446236�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_1446238dense_1_1446240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1446167�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446247�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_1446249dense_2_1446251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1446196|
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1446227*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_1446238*
_output_shapes

:@ *
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameAge_xf:OK
'
_output_shapes
:���������
 
_user_specified_nameSex_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameChest pain type_xf:NJ
'
_output_shapes
:���������

_user_specified_nameBP_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameCholesterol_xf:XT
'
_output_shapes
:���������
)
_user_specified_nameFBS over 120_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameEKG results_xf:RN
'
_output_shapes
:���������
#
_user_specified_name	Max HR_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameExercise angina_xf:Y	U
'
_output_shapes
:���������
*
_user_specified_nameST depression_xf:W
S
'
_output_shapes
:���������
(
_user_specified_nameSlope of ST_xf:c_
'
_output_shapes
:���������
4
_user_specified_nameNumber of vessels fluro_xf:TP
'
_output_shapes
:���������
%
_user_specified_nameThallium_xf:'#
!
_user_specified_name	1446227:'#
!
_user_specified_name	1446229:'#
!
_user_specified_name	1446238:'#
!
_user_specified_name	1446240:'#
!
_user_specified_name	1446249:'#
!
_user_specified_name	1446251
�
�
)__inference_dense_1_layer_call_fn_1446498

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1446167o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:'#
!
_user_specified_name	1446492:'#
!
_user_specified_name	1446494
�;
�
U__inference_transform_features_layer_layer_call_and_return_conditional_losses_1445993
age	
bp	
placeholder	
cholesterol	
placeholder_1	
placeholder_2	
placeholder_3	
placeholder_4	
placeholder_5	
placeholder_6
sex	
placeholder_7	
thallium	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12F
ShapeShapeage*
T0	*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
Shape_1Shapeage*
T0	*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:����������
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
PartitionedCallPartitionedCallagebpplaceholdercholesterolplaceholder_1placeholder_2placeholder_3PlaceholderWithDefault:output:0placeholder_4placeholder_5placeholder_6sexplaceholder_7thalliumunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*3
Tin,
*2(													*
Tout
2	*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_pruned_1445457`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:3*
T0*'
_output_shapes
:���������b

Identity_4IdentityPartitionedCall:output:4*
T0*'
_output_shapes
:���������b

Identity_5IdentityPartitionedCall:output:5*
T0*'
_output_shapes
:���������b

Identity_6IdentityPartitionedCall:output:6*
T0*'
_output_shapes
:���������b

Identity_7IdentityPartitionedCall:output:8*
T0*'
_output_shapes
:���������b

Identity_8IdentityPartitionedCall:output:9*
T0*'
_output_shapes
:���������c

Identity_9IdentityPartitionedCall:output:10*
T0*'
_output_shapes
:���������d
Identity_10IdentityPartitionedCall:output:11*
T0*'
_output_shapes
:���������d
Identity_11IdentityPartitionedCall:output:12*
T0*'
_output_shapes
:���������d
Identity_12IdentityPartitionedCall:output:13*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : :L H
'
_output_shapes
:���������

_user_specified_nameAge:KG
'
_output_shapes
:���������

_user_specified_nameBP:XT
'
_output_shapes
:���������
)
_user_specified_nameChest pain type:TP
'
_output_shapes
:���������
%
_user_specified_nameCholesterol:TP
'
_output_shapes
:���������
%
_user_specified_nameEKG results:XT
'
_output_shapes
:���������
)
_user_specified_nameExercise angina:UQ
'
_output_shapes
:���������
&
_user_specified_nameFBS over 120:OK
'
_output_shapes
:���������
 
_user_specified_nameMax HR:`\
'
_output_shapes
:���������
1
_user_specified_nameNumber of vessels fluro:V	R
'
_output_shapes
:���������
'
_user_specified_nameST depression:L
H
'
_output_shapes
:���������

_user_specified_nameSex:TP
'
_output_shapes
:���������
%
_user_specified_nameSlope of ST:QM
'
_output_shapes
:���������
"
_user_specified_name
Thallium:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: 
�
�
B__inference_dense_layer_call_and_return_conditional_losses_1446462

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
H__inference_concatenate_layer_call_and_return_conditional_losses_1446438
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:���������
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:���������
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:���������
#
_user_specified_name	inputs_12
��
�

__inference_pruned_1445457

inputs	
inputs_1	
inputs_2	
inputs_3	
inputs_4	
inputs_5	
inputs_6	
inputs_7	
inputs_8	
inputs_9	
	inputs_10
	inputs_11	
	inputs_12	
	inputs_13	$
 scale_to_0_1_min_and_max_sub_1_y
scale_to_0_1_less_y&
"scale_to_0_1_1_min_and_max_sub_1_y
scale_to_0_1_1_less_y&
"scale_to_0_1_2_min_and_max_sub_1_y
scale_to_0_1_2_less_y&
"scale_to_0_1_3_min_and_max_sub_1_y
scale_to_0_1_3_less_y&
"scale_to_0_1_4_min_and_max_sub_1_y
scale_to_0_1_4_less_y&
"scale_to_0_1_5_min_and_max_sub_1_y
scale_to_0_1_5_less_y&
"scale_to_0_1_6_min_and_max_sub_1_y
scale_to_0_1_6_less_y&
"scale_to_0_1_7_min_and_max_sub_1_y
scale_to_0_1_7_less_y&
"scale_to_0_1_8_min_and_max_sub_1_y
scale_to_0_1_8_less_y&
"scale_to_0_1_9_min_and_max_sub_1_y
scale_to_0_1_9_less_y'
#scale_to_0_1_10_min_and_max_sub_1_y
scale_to_0_1_10_less_y'
#scale_to_0_1_11_min_and_max_sub_1_y
scale_to_0_1_11_less_y'
#scale_to_0_1_12_min_and_max_sub_1_y
scale_to_0_1_12_less_y
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7	

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13e
 scale_to_0_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    W
scale_to_0_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
scale_to_0_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_3/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_3/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_2/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_4/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_4/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_6/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_6/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_6/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_8/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_8/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_8/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_5/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_7/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_7/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_7/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#scale_to_0_1_11/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
scale_to_0_1_11/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
scale_to_0_1_11/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_9/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_9/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_9/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"scale_to_0_1_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
scale_to_0_1_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#scale_to_0_1_10/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
scale_to_0_1_10/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
scale_to_0_1_10/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#scale_to_0_1_12/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
scale_to_0_1_12/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
scale_to_0_1_12/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
inputs_copyIdentityinputs*
T0	*'
_output_shapes
:���������p
scale_to_0_1/CastCastinputs_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
scale_to_0_1/min_and_max/sub_1Sub)scale_to_0_1/min_and_max/sub_1/x:output:0 scale_to_0_1_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1/subSubscale_to_0_1/Cast:y:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1/zeros_like	ZerosLikescale_to_0_1/sub:z:0*
T0*'
_output_shapes
:���������s
scale_to_0_1/LessLess"scale_to_0_1/min_and_max/sub_1:z:0scale_to_0_1_less_y*
T0*
_output_shapes
: b
scale_to_0_1/Cast_1Castscale_to_0_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1/addAddV2scale_to_0_1/zeros_like:y:0scale_to_0_1/Cast_1:y:0*
T0*'
_output_shapes
:���������r
scale_to_0_1/Cast_2Castscale_to_0_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������s
scale_to_0_1/sub_1Subscale_to_0_1_less_y"scale_to_0_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1/truedivRealDivscale_to_0_1/sub:z:0scale_to_0_1/sub_1:z:0*
T0*'
_output_shapes
:���������h
scale_to_0_1/SigmoidSigmoidscale_to_0_1/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1/SelectV2SelectV2scale_to_0_1/Cast_2:y:0scale_to_0_1/truediv:z:0scale_to_0_1/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1/mulMulscale_to_0_1/SelectV2:output:0scale_to_0_1/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1/add_1AddV2scale_to_0_1/mul:z:0scale_to_0_1/add_1/y:output:0*
T0*'
_output_shapes
:���������^
IdentityIdentityscale_to_0_1/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_1_copyIdentityinputs_1*
T0	*'
_output_shapes
:���������t
scale_to_0_1_3/CastCastinputs_1_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_3/min_and_max/sub_1Sub+scale_to_0_1_3/min_and_max/sub_1/x:output:0"scale_to_0_1_3_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_3/subSubscale_to_0_1_3/Cast:y:0$scale_to_0_1_3/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_3/zeros_like	ZerosLikescale_to_0_1_3/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_3/LessLess$scale_to_0_1_3/min_and_max/sub_1:z:0scale_to_0_1_3_less_y*
T0*
_output_shapes
: f
scale_to_0_1_3/Cast_1Castscale_to_0_1_3/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_3/addAddV2scale_to_0_1_3/zeros_like:y:0scale_to_0_1_3/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_3/Cast_2Castscale_to_0_1_3/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_3/sub_1Subscale_to_0_1_3_less_y$scale_to_0_1_3/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_3/truedivRealDivscale_to_0_1_3/sub:z:0scale_to_0_1_3/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_3/SigmoidSigmoidscale_to_0_1_3/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_3/SelectV2SelectV2scale_to_0_1_3/Cast_2:y:0scale_to_0_1_3/truediv:z:0scale_to_0_1_3/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_3/mulMul scale_to_0_1_3/SelectV2:output:0scale_to_0_1_3/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_3/add_1AddV2scale_to_0_1_3/mul:z:0scale_to_0_1_3/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_1Identityscale_to_0_1_3/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_2_copyIdentityinputs_2*
T0	*'
_output_shapes
:���������t
scale_to_0_1_2/CastCastinputs_2_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_2/min_and_max/sub_1Sub+scale_to_0_1_2/min_and_max/sub_1/x:output:0"scale_to_0_1_2_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_2/subSubscale_to_0_1_2/Cast:y:0$scale_to_0_1_2/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_2/zeros_like	ZerosLikescale_to_0_1_2/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_2/LessLess$scale_to_0_1_2/min_and_max/sub_1:z:0scale_to_0_1_2_less_y*
T0*
_output_shapes
: f
scale_to_0_1_2/Cast_1Castscale_to_0_1_2/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_2/addAddV2scale_to_0_1_2/zeros_like:y:0scale_to_0_1_2/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_2/Cast_2Castscale_to_0_1_2/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_2/sub_1Subscale_to_0_1_2_less_y$scale_to_0_1_2/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_2/truedivRealDivscale_to_0_1_2/sub:z:0scale_to_0_1_2/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_2/SigmoidSigmoidscale_to_0_1_2/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_2/SelectV2SelectV2scale_to_0_1_2/Cast_2:y:0scale_to_0_1_2/truediv:z:0scale_to_0_1_2/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_2/mulMul scale_to_0_1_2/SelectV2:output:0scale_to_0_1_2/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_2/add_1AddV2scale_to_0_1_2/mul:z:0scale_to_0_1_2/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_2Identityscale_to_0_1_2/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_3_copyIdentityinputs_3*
T0	*'
_output_shapes
:���������t
scale_to_0_1_4/CastCastinputs_3_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_4/min_and_max/sub_1Sub+scale_to_0_1_4/min_and_max/sub_1/x:output:0"scale_to_0_1_4_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_4/subSubscale_to_0_1_4/Cast:y:0$scale_to_0_1_4/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_4/zeros_like	ZerosLikescale_to_0_1_4/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_4/LessLess$scale_to_0_1_4/min_and_max/sub_1:z:0scale_to_0_1_4_less_y*
T0*
_output_shapes
: f
scale_to_0_1_4/Cast_1Castscale_to_0_1_4/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_4/addAddV2scale_to_0_1_4/zeros_like:y:0scale_to_0_1_4/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_4/Cast_2Castscale_to_0_1_4/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_4/sub_1Subscale_to_0_1_4_less_y$scale_to_0_1_4/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_4/truedivRealDivscale_to_0_1_4/sub:z:0scale_to_0_1_4/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_4/SigmoidSigmoidscale_to_0_1_4/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_4/SelectV2SelectV2scale_to_0_1_4/Cast_2:y:0scale_to_0_1_4/truediv:z:0scale_to_0_1_4/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_4/mulMul scale_to_0_1_4/SelectV2:output:0scale_to_0_1_4/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_4/add_1AddV2scale_to_0_1_4/mul:z:0scale_to_0_1_4/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_3Identityscale_to_0_1_4/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_4_copyIdentityinputs_4*
T0	*'
_output_shapes
:���������t
scale_to_0_1_6/CastCastinputs_4_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_6/min_and_max/sub_1Sub+scale_to_0_1_6/min_and_max/sub_1/x:output:0"scale_to_0_1_6_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_6/subSubscale_to_0_1_6/Cast:y:0$scale_to_0_1_6/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_6/zeros_like	ZerosLikescale_to_0_1_6/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_6/LessLess$scale_to_0_1_6/min_and_max/sub_1:z:0scale_to_0_1_6_less_y*
T0*
_output_shapes
: f
scale_to_0_1_6/Cast_1Castscale_to_0_1_6/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_6/addAddV2scale_to_0_1_6/zeros_like:y:0scale_to_0_1_6/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_6/Cast_2Castscale_to_0_1_6/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_6/sub_1Subscale_to_0_1_6_less_y$scale_to_0_1_6/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_6/truedivRealDivscale_to_0_1_6/sub:z:0scale_to_0_1_6/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_6/SigmoidSigmoidscale_to_0_1_6/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_6/SelectV2SelectV2scale_to_0_1_6/Cast_2:y:0scale_to_0_1_6/truediv:z:0scale_to_0_1_6/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_6/mulMul scale_to_0_1_6/SelectV2:output:0scale_to_0_1_6/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_6/add_1AddV2scale_to_0_1_6/mul:z:0scale_to_0_1_6/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_4Identityscale_to_0_1_6/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_5_copyIdentityinputs_5*
T0	*'
_output_shapes
:���������t
scale_to_0_1_8/CastCastinputs_5_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_8/min_and_max/sub_1Sub+scale_to_0_1_8/min_and_max/sub_1/x:output:0"scale_to_0_1_8_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_8/subSubscale_to_0_1_8/Cast:y:0$scale_to_0_1_8/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_8/zeros_like	ZerosLikescale_to_0_1_8/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_8/LessLess$scale_to_0_1_8/min_and_max/sub_1:z:0scale_to_0_1_8_less_y*
T0*
_output_shapes
: f
scale_to_0_1_8/Cast_1Castscale_to_0_1_8/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_8/addAddV2scale_to_0_1_8/zeros_like:y:0scale_to_0_1_8/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_8/Cast_2Castscale_to_0_1_8/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_8/sub_1Subscale_to_0_1_8_less_y$scale_to_0_1_8/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_8/truedivRealDivscale_to_0_1_8/sub:z:0scale_to_0_1_8/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_8/SigmoidSigmoidscale_to_0_1_8/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_8/SelectV2SelectV2scale_to_0_1_8/Cast_2:y:0scale_to_0_1_8/truediv:z:0scale_to_0_1_8/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_8/mulMul scale_to_0_1_8/SelectV2:output:0scale_to_0_1_8/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_8/add_1AddV2scale_to_0_1_8/mul:z:0scale_to_0_1_8/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_5Identityscale_to_0_1_8/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_6_copyIdentityinputs_6*
T0	*'
_output_shapes
:���������t
scale_to_0_1_5/CastCastinputs_6_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_5/min_and_max/sub_1Sub+scale_to_0_1_5/min_and_max/sub_1/x:output:0"scale_to_0_1_5_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_5/subSubscale_to_0_1_5/Cast:y:0$scale_to_0_1_5/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_5/zeros_like	ZerosLikescale_to_0_1_5/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_5/LessLess$scale_to_0_1_5/min_and_max/sub_1:z:0scale_to_0_1_5_less_y*
T0*
_output_shapes
: f
scale_to_0_1_5/Cast_1Castscale_to_0_1_5/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_5/addAddV2scale_to_0_1_5/zeros_like:y:0scale_to_0_1_5/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_5/Cast_2Castscale_to_0_1_5/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_5/sub_1Subscale_to_0_1_5_less_y$scale_to_0_1_5/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_5/truedivRealDivscale_to_0_1_5/sub:z:0scale_to_0_1_5/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_5/SigmoidSigmoidscale_to_0_1_5/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_5/SelectV2SelectV2scale_to_0_1_5/Cast_2:y:0scale_to_0_1_5/truediv:z:0scale_to_0_1_5/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_5/mulMul scale_to_0_1_5/SelectV2:output:0scale_to_0_1_5/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_5/add_1AddV2scale_to_0_1_5/mul:z:0scale_to_0_1_5/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_6Identityscale_to_0_1_5/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_7_copyIdentityinputs_7*
T0	*'
_output_shapes
:���������`

Identity_7Identityinputs_7_copy:output:0*
T0	*'
_output_shapes
:���������U
inputs_8_copyIdentityinputs_8*
T0	*'
_output_shapes
:���������t
scale_to_0_1_7/CastCastinputs_8_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_7/min_and_max/sub_1Sub+scale_to_0_1_7/min_and_max/sub_1/x:output:0"scale_to_0_1_7_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_7/subSubscale_to_0_1_7/Cast:y:0$scale_to_0_1_7/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_7/zeros_like	ZerosLikescale_to_0_1_7/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_7/LessLess$scale_to_0_1_7/min_and_max/sub_1:z:0scale_to_0_1_7_less_y*
T0*
_output_shapes
: f
scale_to_0_1_7/Cast_1Castscale_to_0_1_7/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_7/addAddV2scale_to_0_1_7/zeros_like:y:0scale_to_0_1_7/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_7/Cast_2Castscale_to_0_1_7/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_7/sub_1Subscale_to_0_1_7_less_y$scale_to_0_1_7/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_7/truedivRealDivscale_to_0_1_7/sub:z:0scale_to_0_1_7/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_7/SigmoidSigmoidscale_to_0_1_7/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_7/SelectV2SelectV2scale_to_0_1_7/Cast_2:y:0scale_to_0_1_7/truediv:z:0scale_to_0_1_7/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_7/mulMul scale_to_0_1_7/SelectV2:output:0scale_to_0_1_7/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_7/add_1AddV2scale_to_0_1_7/mul:z:0scale_to_0_1_7/add_1/y:output:0*
T0*'
_output_shapes
:���������b

Identity_8Identityscale_to_0_1_7/add_1:z:0*
T0*'
_output_shapes
:���������U
inputs_9_copyIdentityinputs_9*
T0	*'
_output_shapes
:���������u
scale_to_0_1_11/CastCastinputs_9_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
!scale_to_0_1_11/min_and_max/sub_1Sub,scale_to_0_1_11/min_and_max/sub_1/x:output:0#scale_to_0_1_11_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_11/subSubscale_to_0_1_11/Cast:y:0%scale_to_0_1_11/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������r
scale_to_0_1_11/zeros_like	ZerosLikescale_to_0_1_11/sub:z:0*
T0*'
_output_shapes
:���������|
scale_to_0_1_11/LessLess%scale_to_0_1_11/min_and_max/sub_1:z:0scale_to_0_1_11_less_y*
T0*
_output_shapes
: h
scale_to_0_1_11/Cast_1Castscale_to_0_1_11/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_11/addAddV2scale_to_0_1_11/zeros_like:y:0scale_to_0_1_11/Cast_1:y:0*
T0*'
_output_shapes
:���������x
scale_to_0_1_11/Cast_2Castscale_to_0_1_11/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������|
scale_to_0_1_11/sub_1Subscale_to_0_1_11_less_y%scale_to_0_1_11/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_11/truedivRealDivscale_to_0_1_11/sub:z:0scale_to_0_1_11/sub_1:z:0*
T0*'
_output_shapes
:���������n
scale_to_0_1_11/SigmoidSigmoidscale_to_0_1_11/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_11/SelectV2SelectV2scale_to_0_1_11/Cast_2:y:0scale_to_0_1_11/truediv:z:0scale_to_0_1_11/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_11/mulMul!scale_to_0_1_11/SelectV2:output:0scale_to_0_1_11/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_11/add_1AddV2scale_to_0_1_11/mul:z:0 scale_to_0_1_11/add_1/y:output:0*
T0*'
_output_shapes
:���������c

Identity_9Identityscale_to_0_1_11/add_1:z:0*
T0*'
_output_shapes
:���������W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:����������
 scale_to_0_1_9/min_and_max/sub_1Sub+scale_to_0_1_9/min_and_max/sub_1/x:output:0"scale_to_0_1_9_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_9/subSubinputs_10_copy:output:0$scale_to_0_1_9/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_9/zeros_like	ZerosLikescale_to_0_1_9/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_9/LessLess$scale_to_0_1_9/min_and_max/sub_1:z:0scale_to_0_1_9_less_y*
T0*
_output_shapes
: d
scale_to_0_1_9/CastCastscale_to_0_1_9/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_9/addAddV2scale_to_0_1_9/zeros_like:y:0scale_to_0_1_9/Cast:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_9/Cast_1Castscale_to_0_1_9/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_9/sub_1Subscale_to_0_1_9_less_y$scale_to_0_1_9/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_9/truedivRealDivscale_to_0_1_9/sub:z:0scale_to_0_1_9/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_9/SigmoidSigmoidinputs_10_copy:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_9/SelectV2SelectV2scale_to_0_1_9/Cast_1:y:0scale_to_0_1_9/truediv:z:0scale_to_0_1_9/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_9/mulMul scale_to_0_1_9/SelectV2:output:0scale_to_0_1_9/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_9/add_1AddV2scale_to_0_1_9/mul:z:0scale_to_0_1_9/add_1/y:output:0*
T0*'
_output_shapes
:���������c
Identity_10Identityscale_to_0_1_9/add_1:z:0*
T0*'
_output_shapes
:���������W
inputs_11_copyIdentity	inputs_11*
T0	*'
_output_shapes
:���������u
scale_to_0_1_1/CastCastinputs_11_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
 scale_to_0_1_1/min_and_max/sub_1Sub+scale_to_0_1_1/min_and_max/sub_1/x:output:0"scale_to_0_1_1_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_1/subSubscale_to_0_1_1/Cast:y:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������p
scale_to_0_1_1/zeros_like	ZerosLikescale_to_0_1_1/sub:z:0*
T0*'
_output_shapes
:���������y
scale_to_0_1_1/LessLess$scale_to_0_1_1/min_and_max/sub_1:z:0scale_to_0_1_1_less_y*
T0*
_output_shapes
: f
scale_to_0_1_1/Cast_1Castscale_to_0_1_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_1/addAddV2scale_to_0_1_1/zeros_like:y:0scale_to_0_1_1/Cast_1:y:0*
T0*'
_output_shapes
:���������v
scale_to_0_1_1/Cast_2Castscale_to_0_1_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������y
scale_to_0_1_1/sub_1Subscale_to_0_1_1_less_y$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_1/truedivRealDivscale_to_0_1_1/sub:z:0scale_to_0_1_1/sub_1:z:0*
T0*'
_output_shapes
:���������l
scale_to_0_1_1/SigmoidSigmoidscale_to_0_1_1/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_1/SelectV2SelectV2scale_to_0_1_1/Cast_2:y:0scale_to_0_1_1/truediv:z:0scale_to_0_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_1/mulMul scale_to_0_1_1/SelectV2:output:0scale_to_0_1_1/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_1/add_1AddV2scale_to_0_1_1/mul:z:0scale_to_0_1_1/add_1/y:output:0*
T0*'
_output_shapes
:���������c
Identity_11Identityscale_to_0_1_1/add_1:z:0*
T0*'
_output_shapes
:���������W
inputs_12_copyIdentity	inputs_12*
T0	*'
_output_shapes
:���������v
scale_to_0_1_10/CastCastinputs_12_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
!scale_to_0_1_10/min_and_max/sub_1Sub,scale_to_0_1_10/min_and_max/sub_1/x:output:0#scale_to_0_1_10_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_10/subSubscale_to_0_1_10/Cast:y:0%scale_to_0_1_10/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������r
scale_to_0_1_10/zeros_like	ZerosLikescale_to_0_1_10/sub:z:0*
T0*'
_output_shapes
:���������|
scale_to_0_1_10/LessLess%scale_to_0_1_10/min_and_max/sub_1:z:0scale_to_0_1_10_less_y*
T0*
_output_shapes
: h
scale_to_0_1_10/Cast_1Castscale_to_0_1_10/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_10/addAddV2scale_to_0_1_10/zeros_like:y:0scale_to_0_1_10/Cast_1:y:0*
T0*'
_output_shapes
:���������x
scale_to_0_1_10/Cast_2Castscale_to_0_1_10/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������|
scale_to_0_1_10/sub_1Subscale_to_0_1_10_less_y%scale_to_0_1_10/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_10/truedivRealDivscale_to_0_1_10/sub:z:0scale_to_0_1_10/sub_1:z:0*
T0*'
_output_shapes
:���������n
scale_to_0_1_10/SigmoidSigmoidscale_to_0_1_10/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_10/SelectV2SelectV2scale_to_0_1_10/Cast_2:y:0scale_to_0_1_10/truediv:z:0scale_to_0_1_10/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_10/mulMul!scale_to_0_1_10/SelectV2:output:0scale_to_0_1_10/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_10/add_1AddV2scale_to_0_1_10/mul:z:0 scale_to_0_1_10/add_1/y:output:0*
T0*'
_output_shapes
:���������d
Identity_12Identityscale_to_0_1_10/add_1:z:0*
T0*'
_output_shapes
:���������W
inputs_13_copyIdentity	inputs_13*
T0	*'
_output_shapes
:���������v
scale_to_0_1_12/CastCastinputs_13_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
!scale_to_0_1_12/min_and_max/sub_1Sub,scale_to_0_1_12/min_and_max/sub_1/x:output:0#scale_to_0_1_12_min_and_max_sub_1_y*
T0*
_output_shapes
: �
scale_to_0_1_12/subSubscale_to_0_1_12/Cast:y:0%scale_to_0_1_12/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:���������r
scale_to_0_1_12/zeros_like	ZerosLikescale_to_0_1_12/sub:z:0*
T0*'
_output_shapes
:���������|
scale_to_0_1_12/LessLess%scale_to_0_1_12/min_and_max/sub_1:z:0scale_to_0_1_12_less_y*
T0*
_output_shapes
: h
scale_to_0_1_12/Cast_1Castscale_to_0_1_12/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_0_1_12/addAddV2scale_to_0_1_12/zeros_like:y:0scale_to_0_1_12/Cast_1:y:0*
T0*'
_output_shapes
:���������x
scale_to_0_1_12/Cast_2Castscale_to_0_1_12/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:���������|
scale_to_0_1_12/sub_1Subscale_to_0_1_12_less_y%scale_to_0_1_12/min_and_max/sub_1:z:0*
T0*
_output_shapes
: �
scale_to_0_1_12/truedivRealDivscale_to_0_1_12/sub:z:0scale_to_0_1_12/sub_1:z:0*
T0*'
_output_shapes
:���������n
scale_to_0_1_12/SigmoidSigmoidscale_to_0_1_12/Cast:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_12/SelectV2SelectV2scale_to_0_1_12/Cast_2:y:0scale_to_0_1_12/truediv:z:0scale_to_0_1_12/Sigmoid:y:0*
T0*'
_output_shapes
:����������
scale_to_0_1_12/mulMul!scale_to_0_1_12/SelectV2:output:0scale_to_0_1_12/mul/y:output:0*
T0*'
_output_shapes
:����������
scale_to_0_1_12/add_1AddV2scale_to_0_1_12/mul:z:0 scale_to_0_1_12/add_1/y:output:0*
T0*'
_output_shapes
:���������d
Identity_13Identityscale_to_0_1_12/add_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : :- )
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-	)
'
_output_shapes
:���������:-
)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_1446560

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
%__inference_signature_wrapper_1445843
examples
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25:@

unknown_26:@

unknown_27:@ 

unknown_28: 

unknown_29: 

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

 *-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_serve_tf_examples_fn_1445773o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_user_specified_name	1445829:'#
!
_user_specified_name	1445831:'#
!
_user_specified_name	1445833:'#
!
_user_specified_name	1445835:'#
!
_user_specified_name	1445837:' #
!
_user_specified_name	1445839
�
�
'__inference_model_layer_call_fn_1446321

age_xf

sex_xf
placeholder	
bp_xf
cholesterol_xf
placeholder_1
placeholder_2
placeholder_3
placeholder_4
placeholder_5
placeholder_6
placeholder_7
thallium_xf
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallage_xfsex_xfplaceholderbp_xfcholesterol_xfplaceholder_1placeholder_2placeholder_3placeholder_4placeholder_5placeholder_6placeholder_7thallium_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1446263o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameAge_xf:OK
'
_output_shapes
:���������
 
_user_specified_nameSex_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameChest pain type_xf:NJ
'
_output_shapes
:���������

_user_specified_nameBP_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameCholesterol_xf:XT
'
_output_shapes
:���������
)
_user_specified_nameFBS over 120_xf:WS
'
_output_shapes
:���������
(
_user_specified_nameEKG results_xf:RN
'
_output_shapes
:���������
#
_user_specified_name	Max HR_xf:[W
'
_output_shapes
:���������
,
_user_specified_nameExercise angina_xf:Y	U
'
_output_shapes
:���������
*
_user_specified_nameST depression_xf:W
S
'
_output_shapes
:���������
(
_user_specified_nameSlope of ST_xf:c_
'
_output_shapes
:���������
4
_user_specified_nameNumber of vessels fluro_xf:TP
'
_output_shapes
:���������
%
_user_specified_nameThallium_xf:'#
!
_user_specified_name	1446307:'#
!
_user_specified_name	1446309:'#
!
_user_specified_name	1446311:'#
!
_user_specified_name	1446313:'#
!
_user_specified_name	1446315:'#
!
_user_specified_name	1446317
�
�
)__inference_dense_2_layer_call_fn_1446549

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1446196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	1446543:'#
!
_user_specified_name	1446545
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_1446167

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
!dense_1/kernel/Regularizer/L2LossL2Loss8dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0*dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
examples-
serving_default_examples:0���������;
outputs0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-0
layer-14
layer-15
layer_with_weights-1
layer-16
layer-17
layer_with_weights-2
layer-18
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
$P _saved_model_loader_tracked_dict"
_tf_keras_model
J
*0
+1
92
:3
H4
I5"
trackable_list_wrapper
J
*0
+1
92
:3
H4
I5"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_0
Ytrace_12�
'__inference_model_layer_call_fn_1446292
'__inference_model_layer_call_fn_1446321�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0zYtrace_1
�
Ztrace_0
[trace_12�
B__inference_model_layer_call_and_return_conditional_losses_1446211
B__inference_model_layer_call_and_return_conditional_losses_1446263�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0z[trace_1
�B�
"__inference__wrapped_model_1445884Age_xfSex_xfChest pain type_xfBP_xfCholesterol_xfFBS over 120_xfEKG results_xf	Max HR_xfExercise angina_xfST depression_xfSlope of ST_xfNumber of vessels fluro_xfThallium_xf"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
\
_variables
]_iterations
^_learning_rate
__index_dict
`
_momentums
a_velocities
b_update_step_xla"
experimentalOptimizer
,
cserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
-__inference_concatenate_layer_call_fn_1446420�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
�
jtrace_02�
H__inference_concatenate_layer_call_and_return_conditional_losses_1446438�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
'__inference_dense_layer_call_fn_1446447�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
�
qtrace_02�
B__inference_dense_layer_call_and_return_conditional_losses_1446462�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
:@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
wtrace_0
xtrace_12�
)__inference_dropout_layer_call_fn_1446467
)__inference_dropout_layer_call_fn_1446472�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0zxtrace_1
�
ytrace_0
ztrace_12�
D__inference_dropout_layer_call_and_return_conditional_losses_1446484
D__inference_dropout_layer_call_and_return_conditional_losses_1446489�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1
"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_1_layer_call_fn_1446498�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_1_layer_call_and_return_conditional_losses_1446513�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :@ 2dense_1/kernel
: 2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_1_layer_call_fn_1446518
+__inference_dropout_1_layer_call_fn_1446523�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446535
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446540�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_2_layer_call_fn_1446549�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_2_layer_call_and_return_conditional_losses_1446560�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 : 2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
:__inference_transform_features_layer_layer_call_fn_1446086�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
U__inference_transform_features_layer_layer_call_and_return_conditional_losses_1445993�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map"
trackable_dict_wrapper
�
�trace_02�
__inference_loss_fn_0_1446568�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_1446576�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_layer_call_fn_1446292Age_xfSex_xfChest pain type_xfBP_xfCholesterol_xfFBS over 120_xfEKG results_xf	Max HR_xfExercise angina_xfST depression_xfSlope of ST_xfNumber of vessels fluro_xfThallium_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1446321Age_xfSex_xfChest pain type_xfBP_xfCholesterol_xfFBS over 120_xfEKG results_xf	Max HR_xfExercise angina_xfST depression_xfSlope of ST_xfNumber of vessels fluro_xfThallium_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1446211Age_xfSex_xfChest pain type_xfBP_xfCholesterol_xfFBS over 120_xfEKG results_xf	Max HR_xfExercise angina_xfST depression_xfSlope of ST_xfNumber of vessels fluro_xfThallium_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1446263Age_xfSex_xfChest pain type_xfBP_xfCholesterol_xfFBS over 120_xfEKG results_xf	Max HR_xfExercise angina_xfST depression_xfSlope of ST_xfNumber of vessels fluro_xfThallium_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
]0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25B�
%__inference_signature_wrapper_1445843examples"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jexamples
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�	capture_8z�	capture_9z�
capture_10z�
capture_11z�
capture_12z�
capture_13z�
capture_14z�
capture_15z�
capture_16z�
capture_17z�
capture_18z�
capture_19z�
capture_20z�
capture_21z�
capture_22z�
capture_23z�
capture_24z�
capture_25
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_concatenate_layer_call_fn_1446420inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_concatenate_layer_call_and_return_conditional_losses_1446438inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_layer_call_fn_1446447inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_layer_call_and_return_conditional_losses_1446462inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dropout_layer_call_fn_1446467inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_layer_call_fn_1446472inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_1446484inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_1446489inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_1_layer_call_fn_1446498inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_1_layer_call_and_return_conditional_losses_1446513inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_1_layer_call_fn_1446518inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_1_layer_call_fn_1446523inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446535inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446540inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_2_layer_call_fn_1446549inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_2_layer_call_and_return_conditional_losses_1446560inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�	
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25B�
:__inference_transform_features_layer_layer_call_fn_1446086AgeBPChest pain typeCholesterolEKG resultsExercise anginaFBS over 120Max HRNumber of vessels fluroST depressionSexSlope of STThallium"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�	capture_8z�	capture_9z�
capture_10z�
capture_11z�
capture_12z�
capture_13z�
capture_14z�
capture_15z�
capture_16z�
capture_17z�
capture_18z�
capture_19z�
capture_20z�
capture_21z�
capture_22z�
capture_23z�
capture_24z�
capture_25
�	
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25B�
U__inference_transform_features_layer_layer_call_and_return_conditional_losses_1445993AgeBPChest pain typeCholesterolEKG resultsExercise anginaFBS over 120Max HRNumber of vessels fluroST depressionSexSlope of STThallium"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�	capture_8z�	capture_9z�
capture_10z�
capture_11z�
capture_12z�
capture_13z�
capture_14z�
capture_15z�
capture_16z�
capture_17z�
capture_18z�
capture_19z�
capture_20z�
capture_21z�
capture_22z�
capture_23z�
capture_24z�
capture_25
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn"
_generic_user_object
�	
�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25B�
__inference_pruned_1445457inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13"�
���
FullArgSpec
args�	
jarg_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�	capture_8z�	capture_9z�
capture_10z�
capture_11z�
capture_12z�
capture_13z�
capture_14z�
capture_15z�
capture_16z�
capture_17z�
capture_18z�
capture_19z�
capture_20z�
capture_21z�
capture_22z�
capture_23z�
capture_24z�
capture_25
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�B�
__inference_loss_fn_0_1446568"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_1446576"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
#:!@2Adam/m/dense/kernel
#:!@2Adam/v/dense/kernel
:@2Adam/m/dense/bias
:@2Adam/v/dense/bias
%:#@ 2Adam/m/dense_1/kernel
%:#@ 2Adam/v/dense_1/kernel
: 2Adam/m/dense_1/bias
: 2Adam/v/dense_1/bias
%:# 2Adam/m/dense_2/kernel
%:# 2Adam/v/dense_2/kernel
:2Adam/m/dense_2/bias
:2Adam/v/dense_2/bias
"J

Const_25jtf.TrackableConstant
"J

Const_24jtf.TrackableConstant
"J

Const_23jtf.TrackableConstant
"J

Const_22jtf.TrackableConstant
"J

Const_21jtf.TrackableConstant
"J

Const_20jtf.TrackableConstant
"J

Const_19jtf.TrackableConstant
"J

Const_18jtf.TrackableConstant
"J

Const_17jtf.TrackableConstant
"J

Const_16jtf.TrackableConstant
"J

Const_15jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
�

�	capture_0
�	capture_1
�	capture_2
�	capture_3
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�	capture_8
�	capture_9
�
capture_10
�
capture_11
�
capture_12
�
capture_13
�
capture_14
�
capture_15
�
capture_16
�
capture_17
�
capture_18
�
capture_19
�
capture_20
�
capture_21
�
capture_22
�
capture_23
�
capture_24
�
capture_25B�
%__inference_signature_wrapper_1445527inputsinputs_1	inputs_10	inputs_11	inputs_12	inputs_13inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 �

kwonlyargs���
jinputs

jinputs_1
j	inputs_10
j	inputs_11
j	inputs_12
j	inputs_13

jinputs_2

jinputs_3

jinputs_4

jinputs_5

jinputs_6

jinputs_7

jinputs_8

jinputs_9
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�	capture_8z�	capture_9z�
capture_10z�
capture_11z�
capture_12z�
capture_13z�
capture_14z�
capture_15z�
capture_16z�
capture_17z�
capture_18z�
capture_19z�
capture_20z�
capture_21z�
capture_22z�
capture_23z�
capture_24z�
capture_25�
"__inference__wrapped_model_1445884�*+9:HI���
���
���
 �
Age_xf���������
 �
Sex_xf���������
,�)
Chest pain type_xf���������
�
BP_xf���������
(�%
Cholesterol_xf���������
)�&
FBS over 120_xf���������
(�%
EKG results_xf���������
#� 
	Max HR_xf���������
,�)
Exercise angina_xf���������
*�'
ST depression_xf���������
(�%
Slope of ST_xf���������
4�1
Number of vessels fluro_xf���������
%�"
Thallium_xf���������
� "1�.
,
dense_2!�
dense_2����������
H__inference_concatenate_layer_call_and_return_conditional_losses_1446438����
���
���
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
"�
inputs_3���������
"�
inputs_4���������
"�
inputs_5���������
"�
inputs_6���������
"�
inputs_7���������
"�
inputs_8���������
"�
inputs_9���������
#� 
	inputs_10���������
#� 
	inputs_11���������
#� 
	inputs_12���������
� ",�)
"�
tensor_0���������
� �
-__inference_concatenate_layer_call_fn_1446420����
���
���
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
"�
inputs_3���������
"�
inputs_4���������
"�
inputs_5���������
"�
inputs_6���������
"�
inputs_7���������
"�
inputs_8���������
"�
inputs_9���������
#� 
	inputs_10���������
#� 
	inputs_11���������
#� 
	inputs_12���������
� "!�
unknown����������
D__inference_dense_1_layer_call_and_return_conditional_losses_1446513c9:/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
)__inference_dense_1_layer_call_fn_1446498X9:/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
D__inference_dense_2_layer_call_and_return_conditional_losses_1446560cHI/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
)__inference_dense_2_layer_call_fn_1446549XHI/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
B__inference_dense_layer_call_and_return_conditional_losses_1446462c*+/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������@
� �
'__inference_dense_layer_call_fn_1446447X*+/�,
%�"
 �
inputs���������
� "!�
unknown���������@�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446535c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
F__inference_dropout_1_layer_call_and_return_conditional_losses_1446540c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
+__inference_dropout_1_layer_call_fn_1446518X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
+__inference_dropout_1_layer_call_fn_1446523X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
D__inference_dropout_layer_call_and_return_conditional_losses_1446484c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
D__inference_dropout_layer_call_and_return_conditional_losses_1446489c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
)__inference_dropout_layer_call_fn_1446467X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
)__inference_dropout_layer_call_fn_1446472X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@E
__inference_loss_fn_0_1446568$*�

� 
� "�
unknown E
__inference_loss_fn_1_1446576$9�

� 
� "�
unknown �
B__inference_model_layer_call_and_return_conditional_losses_1446211�*+9:HI���
���
���
 �
Age_xf���������
 �
Sex_xf���������
,�)
Chest pain type_xf���������
�
BP_xf���������
(�%
Cholesterol_xf���������
)�&
FBS over 120_xf���������
(�%
EKG results_xf���������
#� 
	Max HR_xf���������
,�)
Exercise angina_xf���������
*�'
ST depression_xf���������
(�%
Slope of ST_xf���������
4�1
Number of vessels fluro_xf���������
%�"
Thallium_xf���������
p

 
� ",�)
"�
tensor_0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1446263�*+9:HI���
���
���
 �
Age_xf���������
 �
Sex_xf���������
,�)
Chest pain type_xf���������
�
BP_xf���������
(�%
Cholesterol_xf���������
)�&
FBS over 120_xf���������
(�%
EKG results_xf���������
#� 
	Max HR_xf���������
,�)
Exercise angina_xf���������
*�'
ST depression_xf���������
(�%
Slope of ST_xf���������
4�1
Number of vessels fluro_xf���������
%�"
Thallium_xf���������
p 

 
� ",�)
"�
tensor_0���������
� �
'__inference_model_layer_call_fn_1446292�*+9:HI���
���
���
 �
Age_xf���������
 �
Sex_xf���������
,�)
Chest pain type_xf���������
�
BP_xf���������
(�%
Cholesterol_xf���������
)�&
FBS over 120_xf���������
(�%
EKG results_xf���������
#� 
	Max HR_xf���������
,�)
Exercise angina_xf���������
*�'
ST depression_xf���������
(�%
Slope of ST_xf���������
4�1
Number of vessels fluro_xf���������
%�"
Thallium_xf���������
p

 
� "!�
unknown����������
'__inference_model_layer_call_fn_1446321�*+9:HI���
���
���
 �
Age_xf���������
 �
Sex_xf���������
,�)
Chest pain type_xf���������
�
BP_xf���������
(�%
Cholesterol_xf���������
)�&
FBS over 120_xf���������
(�%
EKG results_xf���������
#� 
	Max HR_xf���������
,�)
Exercise angina_xf���������
*�'
ST depression_xf���������
(�%
Slope of ST_xf���������
4�1
Number of vessels fluro_xf���������
%�"
Thallium_xf���������
p 

 
� "!�
unknown����������
__inference_pruned_1445457�4�����������������������������
���
���
+
Age$�!

inputs_age���������	
)
BP#� 
	inputs_bp���������	
C
Chest pain type0�-
inputs_chest_pain_type���������	
;
Cholesterol,�)
inputs_cholesterol���������	
;
EKG results,�)
inputs_ekg_results���������	
C
Exercise angina0�-
inputs_exercise_angina���������	
=
FBS over 120-�*
inputs_fbs_over_120���������	
?
Heart Disease.�+
inputs_heart_disease���������	
1
Max HR'�$
inputs_max_hr���������	
S
Number of vessels fluro8�5
inputs_number_of_vessels_fluro���������	
?
ST depression.�+
inputs_st_depression���������
+
Sex$�!

inputs_sex���������	
;
Slope of ST,�)
inputs_slope_of_st���������	
5
Thallium)�&
inputs_thallium���������	
� "���
*
Age_xf �
age_xf���������
(
BP_xf�
bp_xf���������
B
Chest pain type_xf,�)
chest_pain_type_xf���������
:
Cholesterol_xf(�%
cholesterol_xf���������
:
EKG results_xf(�%
ekg_results_xf���������
B
Exercise angina_xf,�)
exercise_angina_xf���������
<
FBS over 120_xf)�&
fbs_over_120_xf���������
>
Heart Disease_xf*�'
heart_disease_xf���������	
0
	Max HR_xf#� 
	max_hr_xf���������
R
Number of vessels fluro_xf4�1
number_of_vessels_fluro_xf���������
>
ST depression_xf*�'
st_depression_xf���������
*
Sex_xf �
sex_xf���������
:
Slope of ST_xf(�%
slope_of_st_xf���������
4
Thallium_xf%�"
thallium_xf����������
%__inference_signature_wrapper_1445527�4�����������������������������
� 
���
*
inputs �
inputs���������	
.
inputs_1"�
inputs_1���������	
0
	inputs_10#� 
	inputs_10���������
0
	inputs_11#� 
	inputs_11���������	
0
	inputs_12#� 
	inputs_12���������	
0
	inputs_13#� 
	inputs_13���������	
.
inputs_2"�
inputs_2���������	
.
inputs_3"�
inputs_3���������	
.
inputs_4"�
inputs_4���������	
.
inputs_5"�
inputs_5���������	
.
inputs_6"�
inputs_6���������	
.
inputs_7"�
inputs_7���������	
.
inputs_8"�
inputs_8���������	
.
inputs_9"�
inputs_9���������	"���
*
Age_xf �
age_xf���������
(
BP_xf�
bp_xf���������
B
Chest pain type_xf,�)
chest_pain_type_xf���������
:
Cholesterol_xf(�%
cholesterol_xf���������
:
EKG results_xf(�%
ekg_results_xf���������
B
Exercise angina_xf,�)
exercise_angina_xf���������
<
FBS over 120_xf)�&
fbs_over_120_xf���������
>
Heart Disease_xf*�'
heart_disease_xf���������	
0
	Max HR_xf#� 
	max_hr_xf���������
R
Number of vessels fluro_xf4�1
number_of_vessels_fluro_xf���������
>
ST depression_xf*�'
st_depression_xf���������
*
Sex_xf �
sex_xf���������
:
Slope of ST_xf(�%
slope_of_st_xf���������
4
Thallium_xf%�"
thallium_xf����������
%__inference_signature_wrapper_1445843�:��������������������������*+9:HI9�6
� 
/�,
*
examples�
examples���������"1�.
,
outputs!�
outputs����������
U__inference_transform_features_layer_layer_call_and_return_conditional_losses_1445993�4�����������������������������
���
���
$
Age�
Age���������	
"
BP�
BP���������	
<
Chest pain type)�&
Chest pain type���������	
4
Cholesterol%�"
Cholesterol���������	
4
EKG results%�"
EKG results���������	
<
Exercise angina)�&
Exercise angina���������	
6
FBS over 120&�#
FBS over 120���������	
*
Max HR �
Max HR���������	
L
Number of vessels fluro1�.
Number of vessels fluro���������	
8
ST depression'�$
ST depression���������
$
Sex�
Sex���������	
4
Slope of ST%�"
Slope of ST���������	
.
Thallium"�
Thallium���������	
� "���
���
3
Age_xf)�&
tensor_0_age_xf���������
1
BP_xf(�%
tensor_0_bp_xf���������
K
Chest pain type_xf5�2
tensor_0_chest_pain_type_xf���������
C
Cholesterol_xf1�.
tensor_0_cholesterol_xf���������
C
EKG results_xf1�.
tensor_0_ekg_results_xf���������
K
Exercise angina_xf5�2
tensor_0_exercise_angina_xf���������
E
FBS over 120_xf2�/
tensor_0_fbs_over_120_xf���������
9
	Max HR_xf,�)
tensor_0_max_hr_xf���������
[
Number of vessels fluro_xf=�:
#tensor_0_number_of_vessels_fluro_xf���������
G
ST depression_xf3�0
tensor_0_st_depression_xf���������
3
Sex_xf)�&
tensor_0_sex_xf���������
C
Slope of ST_xf1�.
tensor_0_slope_of_st_xf���������
=
Thallium_xf.�+
tensor_0_thallium_xf���������
� �
:__inference_transform_features_layer_layer_call_fn_1446086�4�����������������������������
���
���
$
Age�
Age���������	
"
BP�
BP���������	
<
Chest pain type)�&
Chest pain type���������	
4
Cholesterol%�"
Cholesterol���������	
4
EKG results%�"
EKG results���������	
<
Exercise angina)�&
Exercise angina���������	
6
FBS over 120&�#
FBS over 120���������	
*
Max HR �
Max HR���������	
L
Number of vessels fluro1�.
Number of vessels fluro���������	
8
ST depression'�$
ST depression���������
$
Sex�
Sex���������	
4
Slope of ST%�"
Slope of ST���������	
.
Thallium"�
Thallium���������	
� "���
*
Age_xf �
age_xf���������
(
BP_xf�
bp_xf���������
B
Chest pain type_xf,�)
chest_pain_type_xf���������
:
Cholesterol_xf(�%
cholesterol_xf���������
:
EKG results_xf(�%
ekg_results_xf���������
B
Exercise angina_xf,�)
exercise_angina_xf���������
<
FBS over 120_xf)�&
fbs_over_120_xf���������
0
	Max HR_xf#� 
	max_hr_xf���������
R
Number of vessels fluro_xf4�1
number_of_vessels_fluro_xf���������
>
ST depression_xf*�'
st_depression_xf���������
*
Sex_xf �
sex_xf���������
:
Slope of ST_xf(�%
slope_of_st_xf���������
4
Thallium_xf%�"
thallium_xf���������