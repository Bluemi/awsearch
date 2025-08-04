import * as $protobuf from "protobufjs";
import Long = require("long");
/** Namespace abgeordnetenmap. */
export namespace abgeordnetenmap {

    /** Properties of a PreviewQuestion. */
    interface IPreviewQuestion {

        /** PreviewQuestion x */
        x?: (number|null);

        /** PreviewQuestion y */
        y?: (number|null);

        /** PreviewQuestion clusterId */
        clusterId?: (number|null);
    }

    /** Represents a PreviewQuestion. */
    class PreviewQuestion implements IPreviewQuestion {

        /**
         * Constructs a new PreviewQuestion.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.IPreviewQuestion);

        /** PreviewQuestion x. */
        public x: number;

        /** PreviewQuestion y. */
        public y: number;

        /** PreviewQuestion clusterId. */
        public clusterId: number;

        /**
         * Creates a new PreviewQuestion instance using the specified properties.
         * @param [properties] Properties to set
         * @returns PreviewQuestion instance
         */
        public static create(properties?: abgeordnetenmap.IPreviewQuestion): abgeordnetenmap.PreviewQuestion;

        /**
         * Encodes the specified PreviewQuestion message. Does not implicitly {@link abgeordnetenmap.PreviewQuestion.verify|verify} messages.
         * @param message PreviewQuestion message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.IPreviewQuestion, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified PreviewQuestion message, length delimited. Does not implicitly {@link abgeordnetenmap.PreviewQuestion.verify|verify} messages.
         * @param message PreviewQuestion message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.IPreviewQuestion, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a PreviewQuestion message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns PreviewQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.PreviewQuestion;

        /**
         * Decodes a PreviewQuestion message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns PreviewQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.PreviewQuestion;

        /**
         * Verifies a PreviewQuestion message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a PreviewQuestion message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns PreviewQuestion
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.PreviewQuestion;

        /**
         * Creates a plain object from a PreviewQuestion message. Also converts values to other types if specified.
         * @param message PreviewQuestion
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.PreviewQuestion, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this PreviewQuestion to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for PreviewQuestion
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }

    /** Properties of a PreviewCluster. */
    interface IPreviewCluster {

        /** PreviewCluster topic */
        topic?: (string|null);

        /** PreviewCluster centerX */
        centerX?: (number|null);

        /** PreviewCluster centerY */
        centerY?: (number|null);
    }

    /** Represents a PreviewCluster. */
    class PreviewCluster implements IPreviewCluster {

        /**
         * Constructs a new PreviewCluster.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.IPreviewCluster);

        /** PreviewCluster topic. */
        public topic: string;

        /** PreviewCluster centerX. */
        public centerX: number;

        /** PreviewCluster centerY. */
        public centerY: number;

        /**
         * Creates a new PreviewCluster instance using the specified properties.
         * @param [properties] Properties to set
         * @returns PreviewCluster instance
         */
        public static create(properties?: abgeordnetenmap.IPreviewCluster): abgeordnetenmap.PreviewCluster;

        /**
         * Encodes the specified PreviewCluster message. Does not implicitly {@link abgeordnetenmap.PreviewCluster.verify|verify} messages.
         * @param message PreviewCluster message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.IPreviewCluster, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified PreviewCluster message, length delimited. Does not implicitly {@link abgeordnetenmap.PreviewCluster.verify|verify} messages.
         * @param message PreviewCluster message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.IPreviewCluster, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a PreviewCluster message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns PreviewCluster
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.PreviewCluster;

        /**
         * Decodes a PreviewCluster message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns PreviewCluster
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.PreviewCluster;

        /**
         * Verifies a PreviewCluster message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a PreviewCluster message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns PreviewCluster
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.PreviewCluster;

        /**
         * Creates a plain object from a PreviewCluster message. Also converts values to other types if specified.
         * @param message PreviewCluster
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.PreviewCluster, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this PreviewCluster to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for PreviewCluster
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }

    /** Properties of a PreviewQuestionBase. */
    interface IPreviewQuestionBase {

        /** PreviewQuestionBase questions */
        questions?: (abgeordnetenmap.IPreviewQuestion[]|null);

        /** PreviewQuestionBase clusters */
        clusters?: (abgeordnetenmap.IPreviewCluster[]|null);
    }

    /** Represents a PreviewQuestionBase. */
    class PreviewQuestionBase implements IPreviewQuestionBase {

        /**
         * Constructs a new PreviewQuestionBase.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.IPreviewQuestionBase);

        /** PreviewQuestionBase questions. */
        public questions: abgeordnetenmap.IPreviewQuestion[];

        /** PreviewQuestionBase clusters. */
        public clusters: abgeordnetenmap.IPreviewCluster[];

        /**
         * Creates a new PreviewQuestionBase instance using the specified properties.
         * @param [properties] Properties to set
         * @returns PreviewQuestionBase instance
         */
        public static create(properties?: abgeordnetenmap.IPreviewQuestionBase): abgeordnetenmap.PreviewQuestionBase;

        /**
         * Encodes the specified PreviewQuestionBase message. Does not implicitly {@link abgeordnetenmap.PreviewQuestionBase.verify|verify} messages.
         * @param message PreviewQuestionBase message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.IPreviewQuestionBase, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified PreviewQuestionBase message, length delimited. Does not implicitly {@link abgeordnetenmap.PreviewQuestionBase.verify|verify} messages.
         * @param message PreviewQuestionBase message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.IPreviewQuestionBase, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a PreviewQuestionBase message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns PreviewQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.PreviewQuestionBase;

        /**
         * Decodes a PreviewQuestionBase message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns PreviewQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.PreviewQuestionBase;

        /**
         * Verifies a PreviewQuestionBase message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a PreviewQuestionBase message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns PreviewQuestionBase
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.PreviewQuestionBase;

        /**
         * Creates a plain object from a PreviewQuestionBase message. Also converts values to other types if specified.
         * @param message PreviewQuestionBase
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.PreviewQuestionBase, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this PreviewQuestionBase to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for PreviewQuestionBase
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }

    /** Properties of a CompleteQuestion. */
    interface ICompleteQuestion {

        /** CompleteQuestion id */
        id?: (number|null);

        /** CompleteQuestion url */
        url?: (string|null);

        /** CompleteQuestion question */
        question?: (string|null);

        /** CompleteQuestion questionDate */
        questionDate?: (string|null);

        /** CompleteQuestion answer */
        answer?: (string|null);

        /** CompleteQuestion answerDate */
        answerDate?: (string|null);
    }

    /** Represents a CompleteQuestion. */
    class CompleteQuestion implements ICompleteQuestion {

        /**
         * Constructs a new CompleteQuestion.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.ICompleteQuestion);

        /** CompleteQuestion id. */
        public id: number;

        /** CompleteQuestion url. */
        public url: string;

        /** CompleteQuestion question. */
        public question: string;

        /** CompleteQuestion questionDate. */
        public questionDate: string;

        /** CompleteQuestion answer. */
        public answer?: (string|null);

        /** CompleteQuestion answerDate. */
        public answerDate?: (string|null);

        /** CompleteQuestion _answer. */
        public _answer?: "answer";

        /** CompleteQuestion _answerDate. */
        public _answerDate?: "answerDate";

        /**
         * Creates a new CompleteQuestion instance using the specified properties.
         * @param [properties] Properties to set
         * @returns CompleteQuestion instance
         */
        public static create(properties?: abgeordnetenmap.ICompleteQuestion): abgeordnetenmap.CompleteQuestion;

        /**
         * Encodes the specified CompleteQuestion message. Does not implicitly {@link abgeordnetenmap.CompleteQuestion.verify|verify} messages.
         * @param message CompleteQuestion message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.ICompleteQuestion, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified CompleteQuestion message, length delimited. Does not implicitly {@link abgeordnetenmap.CompleteQuestion.verify|verify} messages.
         * @param message CompleteQuestion message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.ICompleteQuestion, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a CompleteQuestion message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns CompleteQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.CompleteQuestion;

        /**
         * Decodes a CompleteQuestion message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns CompleteQuestion
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.CompleteQuestion;

        /**
         * Verifies a CompleteQuestion message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a CompleteQuestion message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns CompleteQuestion
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.CompleteQuestion;

        /**
         * Creates a plain object from a CompleteQuestion message. Also converts values to other types if specified.
         * @param message CompleteQuestion
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.CompleteQuestion, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this CompleteQuestion to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for CompleteQuestion
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }

    /** Properties of a CompleteQuestionBase. */
    interface ICompleteQuestionBase {

        /** CompleteQuestionBase questions */
        questions?: (abgeordnetenmap.ICompleteQuestion[]|null);
    }

    /** Represents a CompleteQuestionBase. */
    class CompleteQuestionBase implements ICompleteQuestionBase {

        /**
         * Constructs a new CompleteQuestionBase.
         * @param [properties] Properties to set
         */
        constructor(properties?: abgeordnetenmap.ICompleteQuestionBase);

        /** CompleteQuestionBase questions. */
        public questions: abgeordnetenmap.ICompleteQuestion[];

        /**
         * Creates a new CompleteQuestionBase instance using the specified properties.
         * @param [properties] Properties to set
         * @returns CompleteQuestionBase instance
         */
        public static create(properties?: abgeordnetenmap.ICompleteQuestionBase): abgeordnetenmap.CompleteQuestionBase;

        /**
         * Encodes the specified CompleteQuestionBase message. Does not implicitly {@link abgeordnetenmap.CompleteQuestionBase.verify|verify} messages.
         * @param message CompleteQuestionBase message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: abgeordnetenmap.ICompleteQuestionBase, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified CompleteQuestionBase message, length delimited. Does not implicitly {@link abgeordnetenmap.CompleteQuestionBase.verify|verify} messages.
         * @param message CompleteQuestionBase message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: abgeordnetenmap.ICompleteQuestionBase, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a CompleteQuestionBase message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns CompleteQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): abgeordnetenmap.CompleteQuestionBase;

        /**
         * Decodes a CompleteQuestionBase message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns CompleteQuestionBase
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): abgeordnetenmap.CompleteQuestionBase;

        /**
         * Verifies a CompleteQuestionBase message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a CompleteQuestionBase message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns CompleteQuestionBase
         */
        public static fromObject(object: { [k: string]: any }): abgeordnetenmap.CompleteQuestionBase;

        /**
         * Creates a plain object from a CompleteQuestionBase message. Also converts values to other types if specified.
         * @param message CompleteQuestionBase
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: abgeordnetenmap.CompleteQuestionBase, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this CompleteQuestionBase to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };

        /**
         * Gets the default type url for CompleteQuestionBase
         * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
         * @returns The default type url
         */
        public static getTypeUrl(typeUrlPrefix?: string): string;
    }
}
