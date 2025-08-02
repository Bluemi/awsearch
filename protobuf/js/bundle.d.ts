import * as $protobuf from "protobufjs";
import Long = require("long");
/** Namespace abgeordnetenmap. */
export namespace abgeordnetenmap {

    /** Properties of a Question. */
    interface IQuestion {

        /** Question x */
        x?: (number|null);

        /** Question y */
        y?: (number|null);

        /** Question clusterId */
        clusterId?: (number|null);
    }

    /** Represents a Question. */
    class Question implements IQuestion {

        /**
         * Constructs a new Question.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.IQuestion);

        /** Question x. */
        public x: number;

        /** Question y. */
        public y: number;

        /** Question clusterId. */
        public clusterId: number;

        /**
         * Creates a new Question instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Question instance
         */
        public static create(properties?: abgeordnetenmap.IQuestion): abgeordnetenmap.Question;

        /**
         * Encodes the specified Question message. Does not implicitly {@link abgeordnetenmap.Question.verify|verify} messages.
         * @param message Question message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.IQuestion, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Question message, length delimited. Does not implicitly {@link abgeordnetenmap.Question.verify|verify} messages.
         * @param message Question message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.IQuestion, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Question message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Question
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.Question;

        /**
         * Decodes a Question message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Question
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.Question;

        /**
         * Verifies a Question message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Question message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Question
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.Question;

        /**
         * Creates a plain object from a Question message. Also converts values to other types if specified.
         * @param message Question
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.Question, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Question to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for Question
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }

    /** Properties of a Cluster. */
    interface ICluster {

        /** Cluster topic */
        topic?: (string|null);

        /** Cluster centerX */
        centerX?: (number|null);

        /** Cluster centerY */
        centerY?: (number|null);
    }

    /** Represents a Cluster. */
    class Cluster implements ICluster {

        /**
         * Constructs a new Cluster.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.ICluster);

        /** Cluster topic. */
        public topic: string;

        /** Cluster centerX. */
        public centerX: number;

        /** Cluster centerY. */
        public centerY: number;

        /**
         * Creates a new Cluster instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Cluster instance
         */
        public static create(properties?: abgeordnetenmap.ICluster): abgeordnetenmap.Cluster;

        /**
         * Encodes the specified Cluster message. Does not implicitly {@link abgeordnetenmap.Cluster.verify|verify} messages.
         * @param message Cluster message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.ICluster, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Cluster message, length delimited. Does not implicitly {@link abgeordnetenmap.Cluster.verify|verify} messages.
         * @param message Cluster message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.ICluster, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Cluster message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Cluster
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.Cluster;

        /**
         * Decodes a Cluster message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Cluster
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.Cluster;

        /**
         * Verifies a Cluster message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Cluster message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Cluster
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.Cluster;

        /**
         * Creates a plain object from a Cluster message. Also converts values to other types if specified.
         * @param message Cluster
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.Cluster, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Cluster to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for Cluster
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }

    /** Properties of a QuestionBase. */
    interface IQuestionBase {

        /** QuestionBase questions */
        questions?: (abgeordnetenmap.IQuestion[]|null);

        /** QuestionBase clusters */
        clusters?: (abgeordnetenmap.ICluster[]|null);
    }

    /** Represents a QuestionBase. */
    class QuestionBase implements IQuestionBase {

        /**
         * Constructs a new QuestionBase.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.IQuestionBase);

        /** QuestionBase questions. */
        public questions: abgeordnetenmap.IQuestion[];

        /** QuestionBase clusters. */
        public clusters: abgeordnetenmap.ICluster[];

        /**
         * Creates a new QuestionBase instance using the specified properties.
         * @param [properties] Properties to set
         * @returns QuestionBase instance
         */
        public static create(properties?: abgeordnetenmap.IQuestionBase): abgeordnetenmap.QuestionBase;

        /**
         * Encodes the specified QuestionBase message. Does not implicitly {@link abgeordnetenmap.QuestionBase.verify|verify} messages.
         * @param message QuestionBase message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.IQuestionBase, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified QuestionBase message, length delimited. Does not implicitly {@link abgeordnetenmap.QuestionBase.verify|verify} messages.
         * @param message QuestionBase message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.IQuestionBase, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a QuestionBase message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns QuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.QuestionBase;

        /**
         * Decodes a QuestionBase message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns QuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.QuestionBase;

        /**
         * Verifies a QuestionBase message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a QuestionBase message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns QuestionBase
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.QuestionBase;

        /**
         * Creates a plain object from a QuestionBase message. Also converts values to other types if specified.
         * @param message QuestionBase
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.QuestionBase, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this QuestionBase to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for QuestionBase
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }
}
