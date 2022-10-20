import tensorflow as tf


class CrossNet(object):
    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))
        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            kernels = [
                tf.get_variable(
                    'kernel' + str(i), shape=[dim, 1], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.36, seed=self.seed),
                    trainable=True)
                for i in range(self.layer_num)
            ]
        elif self.parameterization == 'matrix':
            kernels = [
                tf.get_variable(
                    'kernel' + str(i), shape=[dim, dim], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.36, seed=self.seed),
                    trainable=True)
                for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        bias = [
            tf.get_variable(
                name='bias' + str(i), shape=[dim, 1], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0), trainable=True)
            for i in range(self.layer_num)]
        return kernels, bias

    def infer(self, inputs, phase):
        if len(inputs.shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions {}, expect to be 2 dimensions".format(len(inputs.shape)))
        kernels, bias = self.build(input_shape=inputs.get_shape().as_list())
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        tf_print_list = list()
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                # batch_size, dim, 1 * dim, 1 -> batch_size, 1, 1
                xl_w = tf.tensordot(x_l, kernels[i], axes=(1, 0))
                # batch_size, dim, 1 * batch_size, 1, 1 -> batch_size, dim, 1
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = tf.einsum('ij,bjk->bik', kernels[i], x_l)  # W * xi  (bs, dim, 1)
                # tf_print_list.append(tf.print(f"xl_w {i}", phase, xl_w.name, xl_w))
                dot_ = xl_w + bias[i]  # W * xi + b
                # tf_print_list.append(tf.print(f"dot_ {i}", phase, dot_.name, dot_))
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        # kernel_print = [tf.print(f"dcn kernel print {i}", phase, j.name, j) for i, j in enumerate(kernels)]
        # bias_print = [tf.print(f"dcn bias print {i}", phase, j.name, j, summarize=-1) for i, j in enumerate(bias)]
        # with tf.control_dependencies(kernel_print + bias_print + tf_print_list):
        x_l = tf.squeeze(x_l, axis=2)
        return x_l


class InteractingLayer(object):
    def __init__(self, embedding_size, layer_index, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024):
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        self.embedding_size = embedding_size
        self.layer_index = layer_index
        self.build()

    def build(self):
        self.W_Query = tf.get_variable(
            f'query_{self.layer_index}', shape=[self.embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32, initializer=tf.variance_scaling_initializer(scale=1.0, mode="fan_in", distribution="normal", seed=self.seed))
        self.W_key = tf.get_variable(
            f'key_{self.layer_index}', shape=[self.embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32, initializer=tf.variance_scaling_initializer(scale=1.0, mode="fan_in", distribution="normal", seed=self.seed+1))
        self.W_Value = tf.get_variable(
            f'value_{self.layer_index}', shape=[self.embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32, initializer=tf.variance_scaling_initializer(scale=1.0, mode="fan_in", distribution="normal", seed=self.seed+2))
        if self.use_res:
            # he init
            self.W_Res = tf.get_variable(
                name=f'res_{self.layer_index}', shape=[self.embedding_size, self.att_embedding_size * self.head_num],
                dtype=tf.float32, initializer=tf.variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal", seed=self.seed))

    # batch_size, fea_num, embedding_size
    def infer(self, inputs, layer_num, phase):
        # batch_size, fea_num, embedding_size * embedding_size, head_num x att_embedding_size
        # batch_size, fea_num, head_num x att_embedding_size
        log_list = list()
        input_mean, input_variance = tf.nn.moments(inputs, axes=2)
        log_list.append(tf.print(
            f"autoint input dis log: {layer_num} layer, {phase} phase", input_mean.shape,
            tf.reduce_mean(input_mean, axis=0), tf.reduce_mean(input_variance, axis=0),
            summarize=-1))

        log_list.append(tf.print(
            f"autoint input log: {layer_num} layer, {phase} phase", inputs.shape, inputs))
        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))  # None F D*head_num
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))
        query_mean, query_variance = tf.nn.moments(querys, axes=2)
        log_list.append(tf.print(
            f"autoint query dis log: {layer_num} layer, {phase} phase", query_mean.shape,
            tf.reduce_mean(query_mean, axis=0), tf.reduce_mean(query_variance, axis=0),
            summarize=-1))

        # batch_size, fea_num, head_num x att_embedding_size -> [batch_size, fea_num, att_embedding_size] * head_num
        # -> [head_num, batch_size, fea_num, att_embedding_size]
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        log_list.append(tf.print(
            f"autoint query log: {layer_num} layer, {phase} phase", self.W_Query.name, self.W_Query.shape, self.W_Query))
        log_list.append(tf.print(
            f"autoint key log: {layer_num} layer, {phase} phase", self.W_key.name, self.W_key.shape, self.W_key))
        log_list.append(tf.print(
            f"autoint value log: {layer_num} layer, {phase} phase", self.W_Value.name, self.W_Value.shape, self.W_Value))

        # [head_num, batch_size, fea_num, att_embedding_size] -> [head_num, batch_size, fea_num, fea_num]
        inner_product = tf.matmul(querys, keys, transpose_b=True)  # head_num None F F
        log_list.append(tf.print(
            f"autoint inner_product log: {layer_num} layer, {phase} phase", inner_product.shape, inner_product))
        inner_product_mean, inner_product_variance = tf.nn.moments(inner_product, axes=3)
        log_list.append(tf.print(
            f"autoint inner_product dis log: {layer_num} layer, {phase} phase", inner_product_mean.shape,
            tf.reduce_mean(inner_product_mean, axis=1), tf.reduce_mean(inner_product_variance, axis=1),
            summarize=-1))

        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        # [head_num * batch_size, fea_num, fea_num]
        normalized_att_scores = tf.nn.softmax(inner_product)
        log_list.append(tf.print(
            f"autoint log: {layer_num} layer, {phase} phase", normalized_att_scores.shape,
            tf.reduce_mean(normalized_att_scores, axis=1), summarize=-1))
        # [head_num * batch_size, fea_num, fea_num] * [head_num * batch_size, fea_num, att_embedding_size]
        # [head_num * batch_size, fea_num, att_embedding_size]
        # 分布不变
        result = tf.matmul(normalized_att_scores, values)  # head_num None F D

        result_mean, result_variance = tf.nn.moments(result, axes=3)
        log_list.append(tf.print(
            f"autoint result dis log: {layer_num} layer, {phase} phase", result_mean.shape,
            tf.reduce_mean(result_mean, axis=1), tf.reduce_mean(result_variance, axis=1),
            summarize=-1))

        # [head_num * batch_size, fea_num, att_embedding_size] -> [batch_size, fea_num, att_embedding_size] * head_num
        # -> [batch_size, fea_num, att_embedding_size * head_num]
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0)  # None F D*head_num
        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
            result_mean2, result_variance2 = tf.nn.moments(result, axes=2)
            log_list.append(tf.print(
                f"autoint result2 dis log: {layer_num} layer, {phase} phase", result_mean2.shape,
                tf.reduce_mean(result_mean2, axis=0), tf.reduce_mean(result_variance2, axis=0),
                summarize=-1))
        result = tf.nn.relu(result)
        result_mean3, result_variance3 = tf.nn.moments(result, axes=2)
        log_list.append(tf.print(
            f"autoint result3 dis log: {layer_num} layer, {phase} phase", result_mean3.shape,
            tf.reduce_mean(result_mean3, axis=0), tf.reduce_mean(result_variance3, axis=0),
            summarize=-1))
        log_list.append(tf.print(
            f"autoint result log: {layer_num} layer, {phase} phase", result.shape, result))
        with tf.control_dependencies(log_list):
            result = result * 1.0
        # [batch_size, fea_num, att_embedding_size * head_num]
        return result


class CIN(object):

    def __init__(self, layer_size=(117, 117), seed=1024):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.seed = seed

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        field_nums = [int(input_shape[1])]
        filters = []
        bias = []
        for i, size in enumerate(self.layer_size):
            filters.append(
                tf.get_variable(
                    'filter' + str(i), shape=[1, field_nums[-1] * field_nums[0], size], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.36, seed=self.seed + i),
                    trainable=True))
            bias.append(tf.get_variable(
                    'bias' + str(i), shape=[size], dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    trainable=True))
            field_nums.append(size)
        return filters, bias, field_nums

    def infer(self, inputs):
        # [batch_size, field_num, dim]
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions {}, expect to be 3 dimensions".format(len(inputs.shape)))

        filters, bias, field_nums = self.build(inputs.get_shape().as_list())

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        # dim * [batch_size, field_num, 1]
        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            # dim * [batch_size, field_num, 1]
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            # [dim, batch_size, filed_num, field_num]
            # 不同向量同一维两两乘积
            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)
            # [dim, batch_size, filed_num * field_num]
            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[idx]])
            # [batch_size, dim, field_num * field_num]
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
            # [batch_size, dim, field_num2]
            curr_out = tf.nn.conv1d(
                dot_result, filters=filters[idx], stride=1, padding='VALID')
            curr_out = tf.nn.bias_add(curr_out, bias[idx])
            curr_out = tf.nn.relu(curr_out)
            # [batch_size, field_num2， dim]
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            direct_connect = curr_out
            next_hidden = curr_out
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        # [batch_size, field_num, dim]
        result = tf.concat(final_result, axis=1)
        # [batch_size, field_num]
        result = tf.reduce_sum(result, -1, keep_dims=False)

        return result
