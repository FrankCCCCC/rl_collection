��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02unknown8��
u
dense/kernelVarHandleOp*
shape:	�*
shared_namedense/kernel*
dtype0*
_output_shapes
: 
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	�
m

dense/biasVarHandleOp*
shape:�*
shared_name
dense/bias*
dtype0*
_output_shapes
: 
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:�
�
action_outputs/kernelVarHandleOp*
shape:	�*&
shared_nameaction_outputs/kernel*
dtype0*
_output_shapes
: 
�
)action_outputs/kernel/Read/ReadVariableOpReadVariableOpaction_outputs/kernel*
dtype0*
_output_shapes
:	�
~
action_outputs/biasVarHandleOp*
shape:*$
shared_nameaction_outputs/bias*
dtype0*
_output_shapes
: 
w
'action_outputs/bias/Read/ReadVariableOpReadVariableOpaction_outputs/bias*
dtype0*
_output_shapes
:
�
value_output/kernelVarHandleOp*
shape:	�*$
shared_namevalue_output/kernel*
dtype0*
_output_shapes
: 
|
'value_output/kernel/Read/ReadVariableOpReadVariableOpvalue_output/kernel*
dtype0*
_output_shapes
:	�
z
value_output/biasVarHandleOp*
shape:*"
shared_namevalue_output/bias*
dtype0*
_output_shapes
: 
s
%value_output/bias/Read/ReadVariableOpReadVariableOpvalue_output/bias*
dtype0*
_output_shapes
:

NoOpNoOp
�
ConstConst"/device:CPU:0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
R

	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
�
	variables
 metrics
regularization_losses
!layer_regularization_losses
"non_trainable_variables

#layers
trainable_variables
 
 
 
 
�
$metrics

	variables
regularization_losses
%layer_regularization_losses
&non_trainable_variables

'layers
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
(metrics
	variables
regularization_losses
)layer_regularization_losses
*non_trainable_variables

+layers
trainable_variables
a_
VARIABLE_VALUEaction_outputs/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEaction_outputs/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
,metrics
	variables
regularization_losses
-layer_regularization_losses
.non_trainable_variables

/layers
trainable_variables
_]
VARIABLE_VALUEvalue_output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEvalue_output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
0metrics
	variables
regularization_losses
1layer_regularization_losses
2non_trainable_variables

3layers
trainable_variables
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
y
serving_default_inputsPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsdense/kernel
dense/biasvalue_output/kernelvalue_output/biasaction_outputs/kernelaction_outputs/bias*.
_gradient_op_typePartitionedCall-1523866*.
f)R'
%__inference_signature_wrapper_1523795*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
	2*:
_output_shapes(
&:���������:���������
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp)action_outputs/kernel/Read/ReadVariableOp'action_outputs/bias/Read/ReadVariableOp'value_output/kernel/Read/ReadVariableOp%value_output/bias/Read/ReadVariableOpConst*.
_gradient_op_typePartitionedCall-1523895*)
f$R"
 __inference__traced_save_1523894*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin

2*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasaction_outputs/kernelaction_outputs/biasvalue_output/kernelvalue_output/bias*.
_gradient_op_typePartitionedCall-1523926*,
f'R%
#__inference__traced_restore_1523925*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
	2*
_output_shapes
: ��
�
�
B__inference_model_layer_call_and_return_conditional_losses_1523768

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_21
-action_outputs_statefulpartitionedcall_args_11
-action_outputs_statefulpartitionedcall_args_2
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523637*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1523631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:�����������
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523664*R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_1523658*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0-action_outputs_statefulpartitionedcall_args_1-action_outputs_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523692*T
fORM
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523686*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�
�
B__inference_model_layer_call_and_return_conditional_losses_1523721

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_21
-action_outputs_statefulpartitionedcall_args_11
-action_outputs_statefulpartitionedcall_args_2
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523637*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1523631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:�����������
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523664*R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_1523658*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0-action_outputs_statefulpartitionedcall_args_1-action_outputs_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523692*T
fORM
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523686*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1523808

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523686

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
0__inference_action_outputs_layer_call_fn_1523833

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523692*T
fORM
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523686*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
'__inference_dense_layer_call_fn_1523815

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523637*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1523631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:�����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1523631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
#__inference__traced_restore_1523925
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias,
(assignvariableop_2_action_outputs_kernel*
&assignvariableop_3_action_outputs_bias*
&assignvariableop_4_value_output_kernel(
$assignvariableop_5_value_output_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes

2*,
_output_shapes
::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_action_outputs_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_action_outputs_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_value_output_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_value_output_biasIdentity_5:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: �

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2
RestoreV2_1RestoreV2_12
	RestoreV2	RestoreV22(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_5: : : : :+ '
%
_user_specified_namefile_prefix: : 
�	
�
I__inference_value_output_layer_call_and_return_conditional_losses_1523658

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
B__inference_model_layer_call_and_return_conditional_losses_1523705

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_21
-action_outputs_statefulpartitionedcall_args_11
-action_outputs_statefulpartitionedcall_args_2
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523637*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1523631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:�����������
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523664*R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_1523658*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0-action_outputs_statefulpartitionedcall_args_1-action_outputs_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523692*T
fORM
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523686*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�

�
'__inference_model_layer_call_fn_1523780

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-1523769*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1523768*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
	2*:
_output_shapes(
&:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�#
�
"__inference__wrapped_model_1523614

inputs.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource5
1model_value_output_matmul_readvariableop_resource6
2model_value_output_biasadd_readvariableop_resource7
3model_action_outputs_matmul_readvariableop_resource8
4model_action_outputs_biasadd_readvariableop_resource
identity

identity_1��+model/action_outputs/BiasAdd/ReadVariableOp�*model/action_outputs/MatMul/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�)model/value_output/BiasAdd/ReadVariableOp�(model/value_output/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	��
model/dense/MatMulMatMulinputs)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
(model/value_output/MatMul/ReadVariableOpReadVariableOp1model_value_output_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	��
model/value_output/MatMulMatMulmodel/dense/Relu:activations:00model/value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)model/value_output/BiasAdd/ReadVariableOpReadVariableOp2model_value_output_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model/value_output/BiasAddBiasAdd#model/value_output/MatMul:product:01model/value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model/action_outputs/MatMul/ReadVariableOpReadVariableOp3model_action_outputs_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	��
model/action_outputs/MatMulMatMulmodel/dense/Relu:activations:02model/action_outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+model/action_outputs/BiasAdd/ReadVariableOpReadVariableOp4model_action_outputs_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model/action_outputs/BiasAddBiasAdd%model/action_outputs/MatMul:product:03model/action_outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model/action_outputs/SoftmaxSoftmax%model/action_outputs/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity&model/action_outputs/Softmax:softmax:0,^model/action_outputs/BiasAdd/ReadVariableOp+^model/action_outputs/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*^model/value_output/BiasAdd/ReadVariableOp)^model/value_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identity#model/value_output/BiasAdd:output:0,^model/action_outputs/BiasAdd/ReadVariableOp+^model/action_outputs/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*^model/value_output/BiasAdd/ReadVariableOp)^model/value_output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2V
)model/value_output/BiasAdd/ReadVariableOp)model/value_output/BiasAdd/ReadVariableOp2Z
+model/action_outputs/BiasAdd/ReadVariableOp+model/action_outputs/BiasAdd/ReadVariableOp2X
*model/action_outputs/MatMul/ReadVariableOp*model/action_outputs/MatMul/ReadVariableOp2T
(model/value_output/MatMul/ReadVariableOp(model/value_output/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
�
�
B__inference_model_layer_call_and_return_conditional_losses_1523738

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_21
-action_outputs_statefulpartitionedcall_args_11
-action_outputs_statefulpartitionedcall_args_2
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523637*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1523631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:�����������
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523664*R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_1523658*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0-action_outputs_statefulpartitionedcall_args_1-action_outputs_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523692*T
fORM
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523686*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�

�
%__inference_signature_wrapper_1523795

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-1523784*+
f&R$
"__inference__wrapped_model_1523614*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
	2*:
_output_shapes(
&:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�

�
'__inference_model_layer_call_fn_1523750

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-1523739*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1523738*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
	2*:
_output_shapes(
&:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
�	
�
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523826

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
I__inference_value_output_layer_call_and_return_conditional_losses_1523843

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
 __inference__traced_save_1523894
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop4
0savev2_action_outputs_kernel_read_readvariableop2
.savev2_action_outputs_bias_read_readvariableop2
.savev2_value_output_kernel_read_readvariableop0
,savev2_value_output_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_393f143afbba4b9383466f41477a2a40/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:y
SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop0savev2_action_outputs_kernel_read_readvariableop.savev2_action_outputs_bias_read_readvariableop.savev2_value_output_kernel_read_readvariableop,savev2_value_output_bias_read_readvariableop"/device:CPU:0*
dtypes

2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*K
_input_shapes:
8: :	�:�:	�::	�:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : :+ '
%
_user_specified_namefile_prefix: : 
�
�
.__inference_value_output_layer_call_fn_1523850

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-1523664*R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_1523658*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
inputs/
serving_default_inputs:0���������B
action_outputs0
StatefulPartitionedCall:0���������@
value_output0
StatefulPartitionedCall:1���������tensorflow/serving/predict:�v
� 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
*4&call_and_return_all_conditional_losses
5__call__
6_default_save_signature"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 4], "dtype": "float32", "sparse": false, "name": "inputs"}, "name": "inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_outputs", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_outputs", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["inputs", 0, 0]], "output_layers": [["action_outputs", 0, 0], ["value_output", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 4], "dtype": "float32", "sparse": false, "name": "inputs"}, "name": "inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_outputs", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_outputs", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["inputs", 0, 0]], "output_layers": [["action_outputs", 0, 0], ["value_output", 0, 0]]}}}
�

	variables
regularization_losses
trainable_variables
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "inputs", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 4], "config": {"batch_input_shape": [null, 4], "dtype": "float32", "sparse": false, "name": "inputs"}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*9&call_and_return_all_conditional_losses
:__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*;&call_and_return_all_conditional_losses
<__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "action_outputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "action_outputs", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "value_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�
	variables
 metrics
regularization_losses
!layer_regularization_losses
"non_trainable_variables

#layers
trainable_variables
5__call__
6_default_save_signature
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$metrics

	variables
regularization_losses
%layer_regularization_losses
&non_trainable_variables

'layers
trainable_variables
8__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
:	�2dense/kernel
:�2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
(metrics
	variables
regularization_losses
)layer_regularization_losses
*non_trainable_variables

+layers
trainable_variables
:__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
(:&	�2action_outputs/kernel
!:2action_outputs/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
,metrics
	variables
regularization_losses
-layer_regularization_losses
.non_trainable_variables

/layers
trainable_variables
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
&:$	�2value_output/kernel
:2value_output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
0metrics
	variables
regularization_losses
1layer_regularization_losses
2non_trainable_variables

3layers
trainable_variables
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
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
 "
trackable_list_wrapper
�2�
B__inference_model_layer_call_and_return_conditional_losses_1523721
B__inference_model_layer_call_and_return_conditional_losses_1523705�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_model_layer_call_fn_1523750
'__inference_model_layer_call_fn_1523780�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_1523614�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *%�"
 �
inputs���������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
B__inference_dense_layer_call_and_return_conditional_losses_1523808�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
'__inference_dense_layer_call_fn_1523815�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523826�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
0__inference_action_outputs_layer_call_fn_1523833�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
I__inference_value_output_layer_call_and_return_conditional_losses_1523843�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
.__inference_value_output_layer_call_fn_1523850�
���
FullArgSpec
args�
jself
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
annotations� *
 
3B1
%__inference_signature_wrapper_1523795inputs�
B__inference_model_layer_call_and_return_conditional_losses_1523721�7�4
-�*
 �
inputs���������
p 

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1523705�7�4
-�*
 �
inputs���������
p

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
'__inference_model_layer_call_fn_1523750�7�4
-�*
 �
inputs���������
p

 
� "=�:
�
0���������
�
1����������
'__inference_model_layer_call_fn_1523780�7�4
-�*
 �
inputs���������
p 

 
� "=�:
�
0���������
�
1����������
"__inference__wrapped_model_1523614�/�,
%�"
 �
inputs���������
� "w�t
:
action_outputs(�%
action_outputs���������
6
value_output&�#
value_output����������
B__inference_dense_layer_call_and_return_conditional_losses_1523808]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� {
'__inference_dense_layer_call_fn_1523815P/�,
%�"
 �
inputs���������
� "������������
K__inference_action_outputs_layer_call_and_return_conditional_losses_1523826]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
0__inference_action_outputs_layer_call_fn_1523833P0�-
&�#
!�
inputs����������
� "�����������
I__inference_value_output_layer_call_and_return_conditional_losses_1523843]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
.__inference_value_output_layer_call_fn_1523850P0�-
&�#
!�
inputs����������
� "�����������
%__inference_signature_wrapper_1523795�9�6
� 
/�,
*
inputs �
inputs���������"w�t
:
action_outputs(�%
action_outputs���������
6
value_output&�#
value_output���������