/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
import $protobuf from "protobufjs/minimal";

// Common aliases
const $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;

// Exported root namespace
const $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

export const abgeordnetenmap = $root.abgeordnetenmap = (() => {

    /**
     * Namespace abgeordnetenmap.
     * @exports abgeordnetenmap
     * @namespace
     */
    const abgeordnetenmap = {};

    abgeordnetenmap.PreviewQuestion = (function() {

        /**
         * Properties of a PreviewQuestion.
         * @memberof abgeordnetenmap
         * @interface IPreviewQuestion
         * @property {number|null} [x] PreviewQuestion x
         * @property {number|null} [y] PreviewQuestion y
         * @property {number|null} [clusterId] PreviewQuestion clusterId
         */

        /**
         * Constructs a new PreviewQuestion.
         * @memberof abgeordnetenmap
         * @classdesc Represents a PreviewQuestion.
         * @implements IPreviewQuestion
         * @constructor
         * @param {abgeordnetenmap.IPreviewQuestion=} [properties] Properties to set
         */
        function PreviewQuestion(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * PreviewQuestion x.
         * @member {number} x
         * @memberof abgeordnetenmap.PreviewQuestion
         * @instance
         */
        PreviewQuestion.prototype.x = 0;

        /**
         * PreviewQuestion y.
         * @member {number} y
         * @memberof abgeordnetenmap.PreviewQuestion
         * @instance
         */
        PreviewQuestion.prototype.y = 0;

        /**
         * PreviewQuestion clusterId.
         * @member {number} clusterId
         * @memberof abgeordnetenmap.PreviewQuestion
         * @instance
         */
        PreviewQuestion.prototype.clusterId = 0;

        /**
         * Creates a new PreviewQuestion instance using the specified properties.
         * @function create
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {abgeordnetenmap.IPreviewQuestion=} [properties] Properties to set
         * @returns {abgeordnetenmap.PreviewQuestion} PreviewQuestion instance
         */
        PreviewQuestion.create = function create(properties) {
            return new PreviewQuestion(properties);
        };

        /**
         * Encodes the specified PreviewQuestion message. Does not implicitly {@link abgeordnetenmap.PreviewQuestion.verify|verify} messages.
         * @function encode
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {abgeordnetenmap.IPreviewQuestion} message PreviewQuestion message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        PreviewQuestion.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.x != null && Object.hasOwnProperty.call(message, "x"))
                writer.uint32(/* id 1, wireType 5 =*/13).float(message.x);
            if (message.y != null && Object.hasOwnProperty.call(message, "y"))
                writer.uint32(/* id 2, wireType 5 =*/21).float(message.y);
            if (message.clusterId != null && Object.hasOwnProperty.call(message, "clusterId"))
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.clusterId);
            return writer;
        };

        /**
         * Encodes the specified PreviewQuestion message, length delimited. Does not implicitly {@link abgeordnetenmap.PreviewQuestion.verify|verify} messages.
         * @function encodeDelimited
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {abgeordnetenmap.IPreviewQuestion} message PreviewQuestion message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        PreviewQuestion.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a PreviewQuestion message from the specified reader or buffer.
         * @function decode
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {abgeordnetenmap.PreviewQuestion} PreviewQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        PreviewQuestion.decode = function decode(reader, length, error) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.abgeordnetenmap.PreviewQuestion();
            while (reader.pos < end) {
                let tag = reader.uint32();
                if (tag === error)
                    break;
                switch (tag >>> 3) {
                case 1: {
                        message.x = reader.float();
                        break;
                    }
                case 2: {
                        message.y = reader.float();
                        break;
                    }
                case 3: {
                        message.clusterId = reader.int32();
                        break;
                    }
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a PreviewQuestion message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {abgeordnetenmap.PreviewQuestion} PreviewQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        PreviewQuestion.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a PreviewQuestion message.
         * @function verify
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        PreviewQuestion.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.x != null && message.hasOwnProperty("x"))
                if (typeof message.x !== "number")
                    return "x: number expected";
            if (message.y != null && message.hasOwnProperty("y"))
                if (typeof message.y !== "number")
                    return "y: number expected";
            if (message.clusterId != null && message.hasOwnProperty("clusterId"))
                if (!$util.isInteger(message.clusterId))
                    return "clusterId: integer expected";
            return null;
        };

        /**
         * Creates a PreviewQuestion message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {abgeordnetenmap.PreviewQuestion} PreviewQuestion
         */
        PreviewQuestion.fromObject = function fromObject(object) {
            if (object instanceof $root.abgeordnetenmap.PreviewQuestion)
                return object;
            let message = new $root.abgeordnetenmap.PreviewQuestion();
            if (object.x != null)
                message.x = Number(object.x);
            if (object.y != null)
                message.y = Number(object.y);
            if (object.clusterId != null)
                message.clusterId = object.clusterId | 0;
            return message;
        };

        /**
         * Creates a plain object from a PreviewQuestion message. Also converts values to other types if specified.
         * @function toObject
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {abgeordnetenmap.PreviewQuestion} message PreviewQuestion
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        PreviewQuestion.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.x = 0;
                object.y = 0;
                object.clusterId = 0;
            }
            if (message.x != null && message.hasOwnProperty("x"))
                object.x = options.json && !isFinite(message.x) ? String(message.x) : message.x;
            if (message.y != null && message.hasOwnProperty("y"))
                object.y = options.json && !isFinite(message.y) ? String(message.y) : message.y;
            if (message.clusterId != null && message.hasOwnProperty("clusterId"))
                object.clusterId = message.clusterId;
            return object;
        };

        /**
         * Converts this PreviewQuestion to JSON.
         * @function toJSON
         * @memberof abgeordnetenmap.PreviewQuestion
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        PreviewQuestion.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Gets the default type url for PreviewQuestion
         * @function getTypeUrl
         * @memberof abgeordnetenmap.PreviewQuestion
         * @static
         * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns {string} The default type url
         */
        PreviewQuestion.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
            if (typeUrlPrefix === undefined) {
                typeUrlPrefix = "type.googleapis.com";
            }
            return typeUrlPrefix + "/abgeordnetenmap.PreviewQuestion";
        };

        return PreviewQuestion;
    })();

    abgeordnetenmap.PreviewCluster = (function() {

        /**
         * Properties of a PreviewCluster.
         * @memberof abgeordnetenmap
         * @interface IPreviewCluster
         * @property {string|null} [topic] PreviewCluster topic
         * @property {number|null} [centerX] PreviewCluster centerX
         * @property {number|null} [centerY] PreviewCluster centerY
         */

        /**
         * Constructs a new PreviewCluster.
         * @memberof abgeordnetenmap
         * @classdesc Represents a PreviewCluster.
         * @implements IPreviewCluster
         * @constructor
         * @param {abgeordnetenmap.IPreviewCluster=} [properties] Properties to set
         */
        function PreviewCluster(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * PreviewCluster topic.
         * @member {string} topic
         * @memberof abgeordnetenmap.PreviewCluster
         * @instance
         */
        PreviewCluster.prototype.topic = "";

        /**
         * PreviewCluster centerX.
         * @member {number} centerX
         * @memberof abgeordnetenmap.PreviewCluster
         * @instance
         */
        PreviewCluster.prototype.centerX = 0;

        /**
         * PreviewCluster centerY.
         * @member {number} centerY
         * @memberof abgeordnetenmap.PreviewCluster
         * @instance
         */
        PreviewCluster.prototype.centerY = 0;

        /**
         * Creates a new PreviewCluster instance using the specified properties.
         * @function create
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {abgeordnetenmap.IPreviewCluster=} [properties] Properties to set
         * @returns {abgeordnetenmap.PreviewCluster} PreviewCluster instance
         */
        PreviewCluster.create = function create(properties) {
            return new PreviewCluster(properties);
        };

        /**
         * Encodes the specified PreviewCluster message. Does not implicitly {@link abgeordnetenmap.PreviewCluster.verify|verify} messages.
         * @function encode
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {abgeordnetenmap.IPreviewCluster} message PreviewCluster message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        PreviewCluster.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.topic != null && Object.hasOwnProperty.call(message, "topic"))
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.topic);
            if (message.centerX != null && Object.hasOwnProperty.call(message, "centerX"))
                writer.uint32(/* id 2, wireType 5 =*/21).float(message.centerX);
            if (message.centerY != null && Object.hasOwnProperty.call(message, "centerY"))
                writer.uint32(/* id 3, wireType 5 =*/29).float(message.centerY);
            return writer;
        };

        /**
         * Encodes the specified PreviewCluster message, length delimited. Does not implicitly {@link abgeordnetenmap.PreviewCluster.verify|verify} messages.
         * @function encodeDelimited
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {abgeordnetenmap.IPreviewCluster} message PreviewCluster message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        PreviewCluster.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a PreviewCluster message from the specified reader or buffer.
         * @function decode
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {abgeordnetenmap.PreviewCluster} PreviewCluster
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        PreviewCluster.decode = function decode(reader, length, error) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.abgeordnetenmap.PreviewCluster();
            while (reader.pos < end) {
                let tag = reader.uint32();
                if (tag === error)
                    break;
                switch (tag >>> 3) {
                case 1: {
                        message.topic = reader.string();
                        break;
                    }
                case 2: {
                        message.centerX = reader.float();
                        break;
                    }
                case 3: {
                        message.centerY = reader.float();
                        break;
                    }
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a PreviewCluster message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {abgeordnetenmap.PreviewCluster} PreviewCluster
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        PreviewCluster.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a PreviewCluster message.
         * @function verify
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        PreviewCluster.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.topic != null && message.hasOwnProperty("topic"))
                if (!$util.isString(message.topic))
                    return "topic: string expected";
            if (message.centerX != null && message.hasOwnProperty("centerX"))
                if (typeof message.centerX !== "number")
                    return "centerX: number expected";
            if (message.centerY != null && message.hasOwnProperty("centerY"))
                if (typeof message.centerY !== "number")
                    return "centerY: number expected";
            return null;
        };

        /**
         * Creates a PreviewCluster message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {abgeordnetenmap.PreviewCluster} PreviewCluster
         */
        PreviewCluster.fromObject = function fromObject(object) {
            if (object instanceof $root.abgeordnetenmap.PreviewCluster)
                return object;
            let message = new $root.abgeordnetenmap.PreviewCluster();
            if (object.topic != null)
                message.topic = String(object.topic);
            if (object.centerX != null)
                message.centerX = Number(object.centerX);
            if (object.centerY != null)
                message.centerY = Number(object.centerY);
            return message;
        };

        /**
         * Creates a plain object from a PreviewCluster message. Also converts values to other types if specified.
         * @function toObject
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {abgeordnetenmap.PreviewCluster} message PreviewCluster
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        PreviewCluster.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.topic = "";
                object.centerX = 0;
                object.centerY = 0;
            }
            if (message.topic != null && message.hasOwnProperty("topic"))
                object.topic = message.topic;
            if (message.centerX != null && message.hasOwnProperty("centerX"))
                object.centerX = options.json && !isFinite(message.centerX) ? String(message.centerX) : message.centerX;
            if (message.centerY != null && message.hasOwnProperty("centerY"))
                object.centerY = options.json && !isFinite(message.centerY) ? String(message.centerY) : message.centerY;
            return object;
        };

        /**
         * Converts this PreviewCluster to JSON.
         * @function toJSON
         * @memberof abgeordnetenmap.PreviewCluster
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        PreviewCluster.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Gets the default type url for PreviewCluster
         * @function getTypeUrl
         * @memberof abgeordnetenmap.PreviewCluster
         * @static
         * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns {string} The default type url
         */
        PreviewCluster.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
            if (typeUrlPrefix === undefined) {
                typeUrlPrefix = "type.googleapis.com";
            }
            return typeUrlPrefix + "/abgeordnetenmap.PreviewCluster";
        };

        return PreviewCluster;
    })();

    abgeordnetenmap.PreviewQuestionBase = (function() {

        /**
         * Properties of a PreviewQuestionBase.
         * @memberof abgeordnetenmap
         * @interface IPreviewQuestionBase
         * @property {Array.<abgeordnetenmap.IPreviewQuestion>|null} [questions] PreviewQuestionBase questions
         * @property {Array.<abgeordnetenmap.IPreviewCluster>|null} [clusters] PreviewQuestionBase clusters
         */

        /**
         * Constructs a new PreviewQuestionBase.
         * @memberof abgeordnetenmap
         * @classdesc Represents a PreviewQuestionBase.
         * @implements IPreviewQuestionBase
         * @constructor
         * @param {abgeordnetenmap.IPreviewQuestionBase=} [properties] Properties to set
         */
        function PreviewQuestionBase(properties) {
            this.questions = [];
            this.clusters = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * PreviewQuestionBase questions.
         * @member {Array.<abgeordnetenmap.IPreviewQuestion>} questions
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @instance
         */
        PreviewQuestionBase.prototype.questions = $util.emptyArray;

        /**
         * PreviewQuestionBase clusters.
         * @member {Array.<abgeordnetenmap.IPreviewCluster>} clusters
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @instance
         */
        PreviewQuestionBase.prototype.clusters = $util.emptyArray;

        /**
         * Creates a new PreviewQuestionBase instance using the specified properties.
         * @function create
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {abgeordnetenmap.IPreviewQuestionBase=} [properties] Properties to set
         * @returns {abgeordnetenmap.PreviewQuestionBase} PreviewQuestionBase instance
         */
        PreviewQuestionBase.create = function create(properties) {
            return new PreviewQuestionBase(properties);
        };

        /**
         * Encodes the specified PreviewQuestionBase message. Does not implicitly {@link abgeordnetenmap.PreviewQuestionBase.verify|verify} messages.
         * @function encode
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {abgeordnetenmap.IPreviewQuestionBase} message PreviewQuestionBase message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        PreviewQuestionBase.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.questions != null && message.questions.length)
                for (let i = 0; i < message.questions.length; ++i)
                    $root.abgeordnetenmap.PreviewQuestion.encode(message.questions[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            if (message.clusters != null && message.clusters.length)
                for (let i = 0; i < message.clusters.length; ++i)
                    $root.abgeordnetenmap.PreviewCluster.encode(message.clusters[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified PreviewQuestionBase message, length delimited. Does not implicitly {@link abgeordnetenmap.PreviewQuestionBase.verify|verify} messages.
         * @function encodeDelimited
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {abgeordnetenmap.IPreviewQuestionBase} message PreviewQuestionBase message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        PreviewQuestionBase.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a PreviewQuestionBase message from the specified reader or buffer.
         * @function decode
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {abgeordnetenmap.PreviewQuestionBase} PreviewQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        PreviewQuestionBase.decode = function decode(reader, length, error) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.abgeordnetenmap.PreviewQuestionBase();
            while (reader.pos < end) {
                let tag = reader.uint32();
                if (tag === error)
                    break;
                switch (tag >>> 3) {
                case 1: {
                        if (!(message.questions && message.questions.length))
                            message.questions = [];
                        message.questions.push($root.abgeordnetenmap.PreviewQuestion.decode(reader, reader.uint32()));
                        break;
                    }
                case 2: {
                        if (!(message.clusters && message.clusters.length))
                            message.clusters = [];
                        message.clusters.push($root.abgeordnetenmap.PreviewCluster.decode(reader, reader.uint32()));
                        break;
                    }
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a PreviewQuestionBase message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {abgeordnetenmap.PreviewQuestionBase} PreviewQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        PreviewQuestionBase.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a PreviewQuestionBase message.
         * @function verify
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        PreviewQuestionBase.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.questions != null && message.hasOwnProperty("questions")) {
                if (!Array.isArray(message.questions))
                    return "questions: array expected";
                for (let i = 0; i < message.questions.length; ++i) {
                    let error = $root.abgeordnetenmap.PreviewQuestion.verify(message.questions[i]);
                    if (error)
                        return "questions." + error;
                }
            }
            if (message.clusters != null && message.hasOwnProperty("clusters")) {
                if (!Array.isArray(message.clusters))
                    return "clusters: array expected";
                for (let i = 0; i < message.clusters.length; ++i) {
                    let error = $root.abgeordnetenmap.PreviewCluster.verify(message.clusters[i]);
                    if (error)
                        return "clusters." + error;
                }
            }
            return null;
        };

        /**
         * Creates a PreviewQuestionBase message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {abgeordnetenmap.PreviewQuestionBase} PreviewQuestionBase
         */
        PreviewQuestionBase.fromObject = function fromObject(object) {
            if (object instanceof $root.abgeordnetenmap.PreviewQuestionBase)
                return object;
            let message = new $root.abgeordnetenmap.PreviewQuestionBase();
            if (object.questions) {
                if (!Array.isArray(object.questions))
                    throw TypeError(".abgeordnetenmap.PreviewQuestionBase.questions: array expected");
                message.questions = [];
                for (let i = 0; i < object.questions.length; ++i) {
                    if (typeof object.questions[i] !== "object")
                        throw TypeError(".abgeordnetenmap.PreviewQuestionBase.questions: object expected");
                    message.questions[i] = $root.abgeordnetenmap.PreviewQuestion.fromObject(object.questions[i]);
                }
            }
            if (object.clusters) {
                if (!Array.isArray(object.clusters))
                    throw TypeError(".abgeordnetenmap.PreviewQuestionBase.clusters: array expected");
                message.clusters = [];
                for (let i = 0; i < object.clusters.length; ++i) {
                    if (typeof object.clusters[i] !== "object")
                        throw TypeError(".abgeordnetenmap.PreviewQuestionBase.clusters: object expected");
                    message.clusters[i] = $root.abgeordnetenmap.PreviewCluster.fromObject(object.clusters[i]);
                }
            }
            return message;
        };

        /**
         * Creates a plain object from a PreviewQuestionBase message. Also converts values to other types if specified.
         * @function toObject
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {abgeordnetenmap.PreviewQuestionBase} message PreviewQuestionBase
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        PreviewQuestionBase.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.arrays || options.defaults) {
                object.questions = [];
                object.clusters = [];
            }
            if (message.questions && message.questions.length) {
                object.questions = [];
                for (let j = 0; j < message.questions.length; ++j)
                    object.questions[j] = $root.abgeordnetenmap.PreviewQuestion.toObject(message.questions[j], options);
            }
            if (message.clusters && message.clusters.length) {
                object.clusters = [];
                for (let j = 0; j < message.clusters.length; ++j)
                    object.clusters[j] = $root.abgeordnetenmap.PreviewCluster.toObject(message.clusters[j], options);
            }
            return object;
        };

        /**
         * Converts this PreviewQuestionBase to JSON.
         * @function toJSON
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        PreviewQuestionBase.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Gets the default type url for PreviewQuestionBase
         * @function getTypeUrl
         * @memberof abgeordnetenmap.PreviewQuestionBase
         * @static
         * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns {string} The default type url
         */
        PreviewQuestionBase.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
            if (typeUrlPrefix === undefined) {
                typeUrlPrefix = "type.googleapis.com";
            }
            return typeUrlPrefix + "/abgeordnetenmap.PreviewQuestionBase";
        };

        return PreviewQuestionBase;
    })();

    abgeordnetenmap.CompleteQuestion = (function() {

        /**
         * Properties of a CompleteQuestion.
         * @memberof abgeordnetenmap
         * @interface ICompleteQuestion
         * @property {number|null} [id] CompleteQuestion id
         * @property {string|null} [url] CompleteQuestion url
         * @property {string|null} [question] CompleteQuestion question
         * @property {string|null} [questionDate] CompleteQuestion questionDate
         * @property {string|null} [answer] CompleteQuestion answer
         * @property {string|null} [answerDate] CompleteQuestion answerDate
         */

        /**
         * Constructs a new CompleteQuestion.
         * @memberof abgeordnetenmap
         * @classdesc Represents a CompleteQuestion.
         * @implements ICompleteQuestion
         * @constructor
         * @param {abgeordnetenmap.ICompleteQuestion=} [properties] Properties to set
         */
        function CompleteQuestion(properties) {
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * CompleteQuestion id.
         * @member {number} id
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        CompleteQuestion.prototype.id = 0;

        /**
         * CompleteQuestion url.
         * @member {string} url
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        CompleteQuestion.prototype.url = "";

        /**
         * CompleteQuestion question.
         * @member {string} question
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        CompleteQuestion.prototype.question = "";

        /**
         * CompleteQuestion questionDate.
         * @member {string} questionDate
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        CompleteQuestion.prototype.questionDate = "";

        /**
         * CompleteQuestion answer.
         * @member {string|null|undefined} answer
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        CompleteQuestion.prototype.answer = null;

        /**
         * CompleteQuestion answerDate.
         * @member {string|null|undefined} answerDate
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        CompleteQuestion.prototype.answerDate = null;

        // OneOf field names bound to virtual getters and setters
        let $oneOfFields;

        /**
         * CompleteQuestion _answer.
         * @member {"answer"|undefined} _answer
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        Object.defineProperty(CompleteQuestion.prototype, "_answer", {
            get: $util.oneOfGetter($oneOfFields = ["answer"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        /**
         * CompleteQuestion _answerDate.
         * @member {"answerDate"|undefined} _answerDate
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         */
        Object.defineProperty(CompleteQuestion.prototype, "_answerDate", {
            get: $util.oneOfGetter($oneOfFields = ["answerDate"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        /**
         * Creates a new CompleteQuestion instance using the specified properties.
         * @function create
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {abgeordnetenmap.ICompleteQuestion=} [properties] Properties to set
         * @returns {abgeordnetenmap.CompleteQuestion} CompleteQuestion instance
         */
        CompleteQuestion.create = function create(properties) {
            return new CompleteQuestion(properties);
        };

        /**
         * Encodes the specified CompleteQuestion message. Does not implicitly {@link abgeordnetenmap.CompleteQuestion.verify|verify} messages.
         * @function encode
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {abgeordnetenmap.ICompleteQuestion} message CompleteQuestion message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        CompleteQuestion.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.id);
            if (message.url != null && Object.hasOwnProperty.call(message, "url"))
                writer.uint32(/* id 2, wireType 2 =*/18).string(message.url);
            if (message.question != null && Object.hasOwnProperty.call(message, "question"))
                writer.uint32(/* id 3, wireType 2 =*/26).string(message.question);
            if (message.questionDate != null && Object.hasOwnProperty.call(message, "questionDate"))
                writer.uint32(/* id 4, wireType 2 =*/34).string(message.questionDate);
            if (message.answer != null && Object.hasOwnProperty.call(message, "answer"))
                writer.uint32(/* id 5, wireType 2 =*/42).string(message.answer);
            if (message.answerDate != null && Object.hasOwnProperty.call(message, "answerDate"))
                writer.uint32(/* id 6, wireType 2 =*/50).string(message.answerDate);
            return writer;
        };

        /**
         * Encodes the specified CompleteQuestion message, length delimited. Does not implicitly {@link abgeordnetenmap.CompleteQuestion.verify|verify} messages.
         * @function encodeDelimited
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {abgeordnetenmap.ICompleteQuestion} message CompleteQuestion message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        CompleteQuestion.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a CompleteQuestion message from the specified reader or buffer.
         * @function decode
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {abgeordnetenmap.CompleteQuestion} CompleteQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        CompleteQuestion.decode = function decode(reader, length, error) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.abgeordnetenmap.CompleteQuestion();
            while (reader.pos < end) {
                let tag = reader.uint32();
                if (tag === error)
                    break;
                switch (tag >>> 3) {
                case 1: {
                        message.id = reader.int32();
                        break;
                    }
                case 2: {
                        message.url = reader.string();
                        break;
                    }
                case 3: {
                        message.question = reader.string();
                        break;
                    }
                case 4: {
                        message.questionDate = reader.string();
                        break;
                    }
                case 5: {
                        message.answer = reader.string();
                        break;
                    }
                case 6: {
                        message.answerDate = reader.string();
                        break;
                    }
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a CompleteQuestion message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {abgeordnetenmap.CompleteQuestion} CompleteQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        CompleteQuestion.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a CompleteQuestion message.
         * @function verify
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        CompleteQuestion.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            let properties = {};
            if (message.id != null && message.hasOwnProperty("id"))
                if (!$util.isInteger(message.id))
                    return "id: integer expected";
            if (message.url != null && message.hasOwnProperty("url"))
                if (!$util.isString(message.url))
                    return "url: string expected";
            if (message.question != null && message.hasOwnProperty("question"))
                if (!$util.isString(message.question))
                    return "question: string expected";
            if (message.questionDate != null && message.hasOwnProperty("questionDate"))
                if (!$util.isString(message.questionDate))
                    return "questionDate: string expected";
            if (message.answer != null && message.hasOwnProperty("answer")) {
                properties._answer = 1;
                if (!$util.isString(message.answer))
                    return "answer: string expected";
            }
            if (message.answerDate != null && message.hasOwnProperty("answerDate")) {
                properties._answerDate = 1;
                if (!$util.isString(message.answerDate))
                    return "answerDate: string expected";
            }
            return null;
        };

        /**
         * Creates a CompleteQuestion message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {abgeordnetenmap.CompleteQuestion} CompleteQuestion
         */
        CompleteQuestion.fromObject = function fromObject(object) {
            if (object instanceof $root.abgeordnetenmap.CompleteQuestion)
                return object;
            let message = new $root.abgeordnetenmap.CompleteQuestion();
            if (object.id != null)
                message.id = object.id | 0;
            if (object.url != null)
                message.url = String(object.url);
            if (object.question != null)
                message.question = String(object.question);
            if (object.questionDate != null)
                message.questionDate = String(object.questionDate);
            if (object.answer != null)
                message.answer = String(object.answer);
            if (object.answerDate != null)
                message.answerDate = String(object.answerDate);
            return message;
        };

        /**
         * Creates a plain object from a CompleteQuestion message. Also converts values to other types if specified.
         * @function toObject
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {abgeordnetenmap.CompleteQuestion} message CompleteQuestion
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        CompleteQuestion.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.defaults) {
                object.id = 0;
                object.url = "";
                object.question = "";
                object.questionDate = "";
            }
            if (message.id != null && message.hasOwnProperty("id"))
                object.id = message.id;
            if (message.url != null && message.hasOwnProperty("url"))
                object.url = message.url;
            if (message.question != null && message.hasOwnProperty("question"))
                object.question = message.question;
            if (message.questionDate != null && message.hasOwnProperty("questionDate"))
                object.questionDate = message.questionDate;
            if (message.answer != null && message.hasOwnProperty("answer")) {
                object.answer = message.answer;
                if (options.oneofs)
                    object._answer = "answer";
            }
            if (message.answerDate != null && message.hasOwnProperty("answerDate")) {
                object.answerDate = message.answerDate;
                if (options.oneofs)
                    object._answerDate = "answerDate";
            }
            return object;
        };

        /**
         * Converts this CompleteQuestion to JSON.
         * @function toJSON
         * @memberof abgeordnetenmap.CompleteQuestion
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        CompleteQuestion.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Gets the default type url for CompleteQuestion
         * @function getTypeUrl
         * @memberof abgeordnetenmap.CompleteQuestion
         * @static
         * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns {string} The default type url
         */
        CompleteQuestion.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
            if (typeUrlPrefix === undefined) {
                typeUrlPrefix = "type.googleapis.com";
            }
            return typeUrlPrefix + "/abgeordnetenmap.CompleteQuestion";
        };

        return CompleteQuestion;
    })();

    abgeordnetenmap.CompleteQuestionBase = (function() {

        /**
         * Properties of a CompleteQuestionBase.
         * @memberof abgeordnetenmap
         * @interface ICompleteQuestionBase
         * @property {Array.<abgeordnetenmap.ICompleteQuestion>|null} [questions] CompleteQuestionBase questions
         */

        /**
         * Constructs a new CompleteQuestionBase.
         * @memberof abgeordnetenmap
         * @classdesc Represents a CompleteQuestionBase.
         * @implements ICompleteQuestionBase
         * @constructor
         * @param {abgeordnetenmap.ICompleteQuestionBase=} [properties] Properties to set
         */
        function CompleteQuestionBase(properties) {
            this.questions = [];
            if (properties)
                for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                    if (properties[keys[i]] != null)
                        this[keys[i]] = properties[keys[i]];
        }

        /**
         * CompleteQuestionBase questions.
         * @member {Array.<abgeordnetenmap.ICompleteQuestion>} questions
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @instance
         */
        CompleteQuestionBase.prototype.questions = $util.emptyArray;

        /**
         * Creates a new CompleteQuestionBase instance using the specified properties.
         * @function create
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {abgeordnetenmap.ICompleteQuestionBase=} [properties] Properties to set
         * @returns {abgeordnetenmap.CompleteQuestionBase} CompleteQuestionBase instance
         */
        CompleteQuestionBase.create = function create(properties) {
            return new CompleteQuestionBase(properties);
        };

        /**
         * Encodes the specified CompleteQuestionBase message. Does not implicitly {@link abgeordnetenmap.CompleteQuestionBase.verify|verify} messages.
         * @function encode
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {abgeordnetenmap.ICompleteQuestionBase} message CompleteQuestionBase message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        CompleteQuestionBase.encode = function encode(message, writer) {
            if (!writer)
                writer = $Writer.create();
            if (message.questions != null && message.questions.length)
                for (let i = 0; i < message.questions.length; ++i)
                    $root.abgeordnetenmap.CompleteQuestion.encode(message.questions[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
            return writer;
        };

        /**
         * Encodes the specified CompleteQuestionBase message, length delimited. Does not implicitly {@link abgeordnetenmap.CompleteQuestionBase.verify|verify} messages.
         * @function encodeDelimited
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {abgeordnetenmap.ICompleteQuestionBase} message CompleteQuestionBase message or plain object to encode
         * @param {$protobuf.Writer} [writer] Writer to encode to
         * @returns {$protobuf.Writer} Writer
         */
        CompleteQuestionBase.encodeDelimited = function encodeDelimited(message, writer) {
            return this.encode(message, writer).ldelim();
        };

        /**
         * Decodes a CompleteQuestionBase message from the specified reader or buffer.
         * @function decode
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @param {number} [length] Message length if known beforehand
         * @returns {abgeordnetenmap.CompleteQuestionBase} CompleteQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        CompleteQuestionBase.decode = function decode(reader, length, error) {
            if (!(reader instanceof $Reader))
                reader = $Reader.create(reader);
            let end = length === undefined ? reader.len : reader.pos + length, message = new $root.abgeordnetenmap.CompleteQuestionBase();
            while (reader.pos < end) {
                let tag = reader.uint32();
                if (tag === error)
                    break;
                switch (tag >>> 3) {
                case 1: {
                        if (!(message.questions && message.questions.length))
                            message.questions = [];
                        message.questions.push($root.abgeordnetenmap.CompleteQuestion.decode(reader, reader.uint32()));
                        break;
                    }
                default:
                    reader.skipType(tag & 7);
                    break;
                }
            }
            return message;
        };

        /**
         * Decodes a CompleteQuestionBase message from the specified reader or buffer, length delimited.
         * @function decodeDelimited
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
         * @returns {abgeordnetenmap.CompleteQuestionBase} CompleteQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        CompleteQuestionBase.decodeDelimited = function decodeDelimited(reader) {
            if (!(reader instanceof $Reader))
                reader = new $Reader(reader);
            return this.decode(reader, reader.uint32());
        };

        /**
         * Verifies a CompleteQuestionBase message.
         * @function verify
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {Object.<string,*>} message Plain object to verify
         * @returns {string|null} `null` if valid, otherwise the reason why it is not
         */
        CompleteQuestionBase.verify = function verify(message) {
            if (typeof message !== "object" || message === null)
                return "object expected";
            if (message.questions != null && message.hasOwnProperty("questions")) {
                if (!Array.isArray(message.questions))
                    return "questions: array expected";
                for (let i = 0; i < message.questions.length; ++i) {
                    let error = $root.abgeordnetenmap.CompleteQuestion.verify(message.questions[i]);
                    if (error)
                        return "questions." + error;
                }
            }
            return null;
        };

        /**
         * Creates a CompleteQuestionBase message from a plain object. Also converts values to their respective internal types.
         * @function fromObject
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {Object.<string,*>} object Plain object
         * @returns {abgeordnetenmap.CompleteQuestionBase} CompleteQuestionBase
         */
        CompleteQuestionBase.fromObject = function fromObject(object) {
            if (object instanceof $root.abgeordnetenmap.CompleteQuestionBase)
                return object;
            let message = new $root.abgeordnetenmap.CompleteQuestionBase();
            if (object.questions) {
                if (!Array.isArray(object.questions))
                    throw TypeError(".abgeordnetenmap.CompleteQuestionBase.questions: array expected");
                message.questions = [];
                for (let i = 0; i < object.questions.length; ++i) {
                    if (typeof object.questions[i] !== "object")
                        throw TypeError(".abgeordnetenmap.CompleteQuestionBase.questions: object expected");
                    message.questions[i] = $root.abgeordnetenmap.CompleteQuestion.fromObject(object.questions[i]);
                }
            }
            return message;
        };

        /**
         * Creates a plain object from a CompleteQuestionBase message. Also converts values to other types if specified.
         * @function toObject
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {abgeordnetenmap.CompleteQuestionBase} message CompleteQuestionBase
         * @param {$protobuf.IConversionOptions} [options] Conversion options
         * @returns {Object.<string,*>} Plain object
         */
        CompleteQuestionBase.toObject = function toObject(message, options) {
            if (!options)
                options = {};
            let object = {};
            if (options.arrays || options.defaults)
                object.questions = [];
            if (message.questions && message.questions.length) {
                object.questions = [];
                for (let j = 0; j < message.questions.length; ++j)
                    object.questions[j] = $root.abgeordnetenmap.CompleteQuestion.toObject(message.questions[j], options);
            }
            return object;
        };

        /**
         * Converts this CompleteQuestionBase to JSON.
         * @function toJSON
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @instance
         * @returns {Object.<string,*>} JSON object
         */
        CompleteQuestionBase.prototype.toJSON = function toJSON() {
            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
        };

        /**
         * Gets the default type url for CompleteQuestionBase
         * @function getTypeUrl
         * @memberof abgeordnetenmap.CompleteQuestionBase
         * @static
         * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns {string} The default type url
         */
        CompleteQuestionBase.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
            if (typeUrlPrefix === undefined) {
                typeUrlPrefix = "type.googleapis.com";
            }
            return typeUrlPrefix + "/abgeordnetenmap.CompleteQuestionBase";
        };

        return CompleteQuestionBase;
    })();

    return abgeordnetenmap;
})();

export { $root as default };
