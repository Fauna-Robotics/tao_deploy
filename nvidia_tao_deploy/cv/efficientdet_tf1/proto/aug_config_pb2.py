# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_deploy/cv/efficientdet_tf1/proto/aug_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_deploy/cv/efficientdet_tf1/proto/aug_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n<nvidia_tao_deploy/cv/efficientdet_tf1/proto/aug_config.proto\"]\n\tAugConfig\x12\x12\n\nrand_hflip\x18\x01 \x01(\x08\x12\x1d\n\x15random_crop_min_scale\x18\x02 \x01(\x02\x12\x1d\n\x15random_crop_max_scale\x18\x03 \x01(\x02\x62\x06proto3')
)




_AUGCONFIG = _descriptor.Descriptor(
  name='AugConfig',
  full_name='AugConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rand_hflip', full_name='AugConfig.rand_hflip', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_crop_min_scale', full_name='AugConfig.random_crop_min_scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_crop_max_scale', full_name='AugConfig.random_crop_max_scale', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=64,
  serialized_end=157,
)

DESCRIPTOR.message_types_by_name['AugConfig'] = _AUGCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AugConfig = _reflection.GeneratedProtocolMessageType('AugConfig', (_message.Message,), dict(
  DESCRIPTOR = _AUGCONFIG,
  __module__ = 'nvidia_tao_deploy.cv.efficientdet_tf1.proto.aug_config_pb2'
  # @@protoc_insertion_point(class_scope:AugConfig)
  ))
_sym_db.RegisterMessage(AugConfig)


# @@protoc_insertion_point(module_scope)
